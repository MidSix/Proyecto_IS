from PyQt5.QtWidgets import QComboBox, QListView

def setup_global_combobox_behavior() -> None:
    """Customize global QComboBox behavior for better UX.

    Modifies QComboBox.__init__ and showPopup methods globally to:
    1. Remove decorations from list items (no selection highlights).
    2. Dynamically size the dropdown popup to fit content width.

    This function should be called once during application startup to
    affect all QComboBox instances created thereafter.

    Returns
    -------
    None
    """
    original_init = QComboBox.__init__

    def custom_init(self, *args, **kwargs) -> None:
        """Initialize QComboBox with custom styling for list view."""
        original_init(self, *args, **kwargs)

        view = QListView()
        view.setStyleSheet("""
            QListView {
                show-decoration-selected: 0;
            }
            QListView::item {
                border: none;
            }
        """)
        self.setView(view)

    QComboBox.__init__ = custom_init

    original_show_popup = QComboBox.showPopup

    def custom_show_popup(self) -> None:
        """Display popup and adjust its width to fit content."""
        original_show_popup(self)

        view = self.view()
        if view is not None:
            text_width = view.sizeHintForColumn(0)
            extra = 6
            target_width = max(self.width(), text_width + extra)
            view.setMinimumWidth(target_width)

    QComboBox.showPopup = custom_show_popup