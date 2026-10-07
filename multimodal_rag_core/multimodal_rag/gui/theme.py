"""Named GUI color themes.

A theme is a flat palette; ``app_qss`` turns it into the application stylesheet
and the panels read individual colors for their inline-styled bubbles/cards.
Adding a theme = add a Palette to ``THEMES`` and a GuiTheme enum member.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    window_bg: str
    surface_bg: str
    card_bg: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_text: str
    accent_hover: str
    accent_pressed: str
    disabled_bg: str
    disabled_text: str
    user_bubble_bg: str
    user_bubble_text: str
    assistant_bubble_bg: str
    assistant_bubble_text: str
    link: str
    badge_bg: str
    badge_text: str
    input_bg: str
    input_border: str
    warn: str


THEMES: dict[str, Palette] = {
    "Midnight": Palette(
        name="Midnight",
        window_bg="#1a1b23", surface_bg="#23242e", card_bg="#2b2d38",
        border="#3a3c48", text="#e3e5e8", text_muted="#9aa0a8",
        accent="#5865f2", accent_text="#ffffff", accent_hover="#6b76ff",
        accent_pressed="#4752c4", disabled_bg="#3a3c48", disabled_text="#7a7d86",
        user_bubble_bg="#5865f2", user_bubble_text="#ffffff",
        assistant_bubble_bg="#2b2d38", assistant_bubble_text="#e3e5e8",
        link="#7ea6ff", badge_bg="#3a3c48", badge_text="#9db2ff",
        input_bg="#1e1f29", input_border="#3a3c48", warn="#f0b232",
    ),
    "Graphite": Palette(
        name="Graphite",
        window_bg="#12121c", surface_bg="#181826", card_bg="#1a1a2e",
        border="#2a2a3e", text="#e0e0e0", text_muted="#a0a0b0",
        accent="#e8a04c", accent_text="#1a1a26", accent_hover="#f0b266",
        accent_pressed="#c9832b", disabled_bg="#3a3a4a", disabled_text="#777788",
        user_bubble_bg="#e8a04c", user_bubble_text="#1a1a26",
        assistant_bubble_bg="#252538", assistant_bubble_text="#e0e0e0",
        link="#f0a860", badge_bg="#3a2d20", badge_text="#f0a860",
        input_bg="#181826", input_border="#2a2a3e", warn="#b8860b",
    ),
    "Ocean": Palette(
        name="Ocean",
        window_bg="#0b1d2a", surface_bg="#0f2838", card_bg="#143246",
        border="#1f4a63", text="#e6f1f7", text_muted="#8fb0c2",
        accent="#f59e0b", accent_text="#1a1205", accent_hover="#fbbf24",
        accent_pressed="#d97706", disabled_bg="#1f4a63", disabled_text="#6f93a8",
        user_bubble_bg="#f59e0b", user_bubble_text="#1a1205",
        assistant_bubble_bg="#143246", assistant_bubble_text="#e6f1f7",
        link="#7dd3fc", badge_bg="#123244", badge_text="#7dd3fc",
        input_bg="#0f2838", input_border="#1f4a63", warn="#fbbf24",
    ),
    # "Sunset" disabled for now.
    # "Sunset": Palette(
    #     name="Sunset",
    #     window_bg="#1c1410", surface_bg="#261b16", card_bg="#32241d",
    #     border="#4e3a2f", text="#f8ede4", text_muted="#c2a894",
    #     accent="#f472b6", accent_text="#2e0f22", accent_hover="#f9a8d4",
    #     accent_pressed="#db5a9e", disabled_bg="#4e3a2f", disabled_text="#92786b",
    #     user_bubble_bg="#f472b6", user_bubble_text="#2e0f22",
    #     assistant_bubble_bg="#32241d", assistant_bubble_text="#f8ede4",
    #     link="#fdba74", badge_bg="#3a2a22", badge_text="#fdba74",
    #     input_bg="#261b16", input_border="#4e3a2f", warn="#fbbf24",
    # ),
    "Daylight": Palette(
        name="Daylight",
        window_bg="#e4e7ea", surface_bg="#eff1f4", card_bg="#e9ecef",
        border="#cfd4db", text="#1f2430", text_muted="#5b626e",
        accent="#2f6fed", accent_text="#ffffff", accent_hover="#4b83f0",
        accent_pressed="#255bd0", disabled_bg="#d3d7dd", disabled_text="#9aa0ab",
        user_bubble_bg="#2f6fed", user_bubble_text="#ffffff",
        assistant_bubble_bg="#ffffff", assistant_bubble_text="#1f2430",
        link="#1d4ed8", badge_bg="#dfe6fb", badge_text="#255bd0",
        input_bg="#ffffff", input_border="#cfd4db", warn="#b45309",
    ),
}

DEFAULT_THEME = "Graphite"


def get_palette(name: str | None) -> Palette:
    return THEMES.get(name or "", THEMES[DEFAULT_THEME])


def app_qss(p: Palette) -> str:
    return f"""
QWidget {{ background-color: {p.window_bg}; color: {p.text}; font-size: 13px; }}
QMainWindow, QMenuBar {{ background-color: {p.window_bg}; color: {p.text}; }}
QMenu {{ background-color: {p.card_bg}; color: {p.text}; border: 1px solid {p.border}; }}
QMenu::item {{ background-color: transparent; color: {p.text}; padding: 6px 24px 6px 12px; }}
QMenu::item:selected {{ background-color: {p.accent}; color: {p.accent_text}; }}
QMenu::item:disabled {{ color: {p.disabled_text}; }}
QMenu::separator {{ height: 1px; background: {p.border}; margin: 4px 8px; }}
QMenuBar::item {{ background: transparent; color: {p.text}; padding: 4px 10px; }}
QMenuBar::item:selected {{ background-color: {p.card_bg}; color: {p.text}; }}
QMenuBar::item:pressed {{ background-color: {p.accent}; color: {p.accent_text}; }}
QToolTip {{ background-color: {p.card_bg}; color: {p.text}; border: 1px solid {p.border}; }}
QLineEdit, QComboBox, QPlainTextEdit {{
    background-color: {p.input_bg}; color: {p.text}; border: 1px solid {p.input_border};
    border-radius: 6px; padding: 6px;
}}
QTextBrowser, QTableWidget {{
    background-color: {p.surface_bg}; color: {p.text}; border: 1px solid {p.input_border};
    border-radius: 6px; padding: 6px;
}}
QPushButton {{
    background-color: {p.accent}; color: {p.accent_text}; border: none; border-radius: 6px;
    padding: 7px 12px; font-weight: bold;
}}
QPushButton:hover {{ background-color: {p.accent_hover}; }}
QPushButton:pressed {{ background-color: {p.accent_pressed}; }}
QPushButton:disabled {{ background-color: {p.disabled_bg}; color: {p.disabled_text}; }}
QHeaderView::section {{ background-color: {p.card_bg}; color: {p.text_muted}; border: none; padding: 5px; }}
QSplitter::handle {{ background-color: {p.border}; }}
QStatusBar {{ background-color: {p.card_bg}; color: {p.text_muted}; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px 2px 2px 0; }}
QScrollBar::groove:vertical {{ background: {p.surface_bg}; border-radius: 6px; }}
QScrollBar::handle:vertical {{ background: {p.text_muted}; border-radius: 6px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {p.text}; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; background: none; border: none; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::groove:horizontal {{ background: {p.surface_bg}; border-radius: 6px; }}
QScrollBar::handle:horizontal {{ background: {p.text_muted}; border-radius: 6px; min-width: 28px; }}
QScrollBar::handle:horizontal:hover {{ background: {p.text}; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; background: none; border: none; }}
"""
