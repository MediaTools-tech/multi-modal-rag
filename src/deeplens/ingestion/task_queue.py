"""Asynchronous ingestion queue and worker loop.

Coordinates reading, chunking, embedding generation, database inserts, and progress callbacks.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Callable, Coroutine

from PIL import Image
import structlog

from deeplens.config import Settings
from deeplens.core.embedding import EmbeddingEngine
from deeplens.core.models import FileType, IngestionJob, IndexingProgress, VectorRecord
from deeplens.core.repository import DocumentRepository
from deeplens.ingestion.router import FileRouter

# Import chunkers
from deeplens.ingestion.chunkers.document import DocumentChunker
from deeplens.ingestion.chunkers.image import ImageChunker
from deeplens.ingestion.chunkers.audio import AudioChunker
from deeplens.ingestion.chunkers.video import VideoChunker
from deeplens.ingestion.chunkers.subtitle import SubtitleChunker

logger = structlog.get_logger(__name__)


class IngestionQueue:
    """Manages the queue of files to be ingested and indexes them concurrently."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_engine: EmbeddingEngine,
        settings: Settings,
        progress_callback: Callable[[IndexingProgress], None] | None = None
    ) -> None:
        self.repository = repository
        self.embedding_engine = embedding_engine
        self.settings = settings
        self.progress_callback = progress_callback
        
        self.queue: asyncio.Queue[IngestionJob] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._active = False
        
        # State tracking for progress
        self._progress_map: dict[str, IndexingProgress] = {}
        self._progress_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the background worker tasks."""
        if self._active:
            return
        
        self._active = True
        num_workers = self.settings.ingestion_workers
        self._workers = [
            asyncio.create_task(self._worker_loop(i)) for i in range(num_workers)
        ]
        logger.info("ingestion_queue.started", workers=num_workers)

    async def stop(self) -> None:
        """Stop queue worker loop and wait for completion."""
        self._active = False
        # Feed None jobs or cancel tasks
        for t in self._workers:
            t.cancel()
        
        # Wait with gather
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers = []
        logger.info("ingestion_queue.stopped")

    async def submit(self, path: Path) -> str:
        """Submit a file or folder for ingestion."""
        path = path.resolve()
        
        if path.is_file():
            job = IngestionJob(source_path=path, file_type=FileRouter.route(path))
            await self.queue.put(job)
            logger.info("ingestion_queue.submitted_file", path=str(path), job_id=job.id)
            return job.id
            
        elif path.is_dir():
            # Scan directory and group files
            grouped = FileRouter.scan_directory(path)
            all_files = []
            for files in grouped.values():
                all_files.extend(files)
            
            # Setup IndexingProgress tracking for this folder scan
            folder_path = str(path)
            async with self._progress_lock:
                self._progress_map[folder_path] = IndexingProgress(
                    folder_path=folder_path,
                    total_files=len(all_files),
                    status="scanning",
                )
                self._trigger_progress(folder_path)

            for fp in all_files:
                job = IngestionJob(source_path=fp, file_type=FileRouter.route(fp))
                await self.queue.put(job)
            
            async with self._progress_lock:
                self._progress_map[folder_path].status = "indexing"
                self._trigger_progress(folder_path)
                
            logger.info("ingestion_queue.submitted_directory", path=str(path), count=len(all_files))
            return folder_path

        return ""

    def _trigger_progress(self, key: str) -> None:
        if self.progress_callback and key in self._progress_map:
            self.progress_callback(self._progress_map[key])

    async def _worker_loop(self, worker_id: int) -> None:
        while self._active:
            try:
                # Dequeue next job
                job = await self.queue.get()
                
                # Update job status
                job.status = "processing"
                logger.info("ingestion_queue.worker.process", worker=worker_id, file=str(job.source_path))

                # Track progress associated with parent folder
                parent_dir = str(job.source_path.parent.resolve())
                
                # Check if we should index (hash check)
                meta = FileRouter.get_file_metadata(job.source_path)
                needs_index = await self.repository.file_needs_reindex(
                    meta["absolute_path"], meta["file_hash"]
                )

                records_created = 0
                if needs_index:
                    # Clean up old records for this path
                    await self.repository.delete_by_path(meta["absolute_path"])
                    # Generate new records
                    records_created = await self._process_job(job)
                    job.records_created = records_created
                    job.status = "completed"
                else:
                    job.status = "completed"  # Skipped, up to date
                    logger.info("ingestion_queue.worker.skip_up_to_date", file=str(job.source_path))

                # Update progress tracking
                await self._update_progress_stats(parent_dir, job.source_path.name, success=True)
                
                job.completed_at = datetime.utcnow().isoformat()
                self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ingestion_queue.worker.failed", worker=worker_id, error=str(e))
                if 'job' in locals():
                    job.status = "failed"
                    job.error = str(e)
                    parent_dir = str(job.source_path.parent.resolve())
                    await self._update_progress_stats(parent_dir, job.source_path.name, success=False)
                    self.queue.task_done()

    async def _update_progress_stats(self, parent_dir: str, file_name: str, success: bool) -> None:
        async with self._progress_lock:
            # Find progress tracker for this parent folder or its parent subtrees
            for key, tracker in self._progress_map.items():
                # If tracker folder_path matches parent_dir or contains it
                if parent_dir.startswith(key):
                    tracker.processed_files += 1
                    tracker.current_file = file_name
                    if not success:
                        tracker.failed_files += 1
                    
                    if tracker.processed_files >= tracker.total_files:
                        tracker.status = "completed"
                    self._trigger_progress(key)

    async def _process_job(self, job: IngestionJob) -> int:
        """Route file to chunker, calculate embeddings, and save to repository."""
        records: list[VectorRecord] = []
        
        # 1. Chunking
        if job.file_type == FileType.DOCUMENT:
            records = await asyncio.to_thread(DocumentChunker.chunk, job.source_path, self.settings)
        elif job.file_type == FileType.IMAGE:
            records = await asyncio.to_thread(ImageChunker.chunk, job.source_path, self.settings)
        elif job.file_type == FileType.AUDIO:
            # transcription requires heavier processing, offloaded to thread
            records = await asyncio.to_thread(AudioChunker.chunk, job.source_path, self.settings)
        elif job.file_type == FileType.VIDEO:
            records = await asyncio.to_thread(VideoChunker.chunk, job.source_path, self.settings)
        elif job.file_type == FileType.SUBTITLE:
            records = await asyncio.to_thread(SubtitleChunker.chunk, job.source_path, self.settings)
        else:
            logger.warn("ingestion_queue.unknown_file_type", file=str(job.source_path))
            return 0

        if not records:
            return 0

        # 2. Embedding Generation
        # For each record, generate vector embedding
        logger.info("ingestion_queue.generate_embeddings", count=len(records))
        
        # Process visual frames or clips paths if they exist in metadata
        for r in records:
            # Parse metadata
            import json
            try:
                m_data = json.loads(r.metadata_json)
            except Exception:
                m_data = {}

            # Local video visual frames:
            if "temp_frame_path" in m_data:
                frame_path = Path(m_data["temp_frame_path"])
                if frame_path.exists():
                    # Generate embedding directly from the image frame
                    r.vector = await self.embedding_engine.embed_image_from_path(frame_path)
                    # Clean up temp frame file
                    try:
                        frame_path.unlink()
                    except Exception:
                        pass
                    # Remove temp path from final DB row metadata
                    m_data.pop("temp_frame_path", None)
                    r.metadata_json = json.dumps(m_data)

            # Cloud video clips or regular images:
            elif r.file_type == FileType.IMAGE.value:
                # Regular image
                r.vector = await self.embedding_engine.embed_image_from_path(Path(r.absolute_path))
            
            else:
                # Text record (documents, subtitles, audio transcripts)
                r.vector = await self.embedding_engine.embed_text(r.content)

        # Remove empty vector records (e.g. if embedding failed)
        valid_records = [r for r in records if r.vector]
        if not valid_records:
            return 0

        # 3. Store in Repository
        inserted = await self.repository.insert(valid_records)
        return inserted
