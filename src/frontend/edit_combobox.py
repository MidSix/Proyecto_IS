from PyQt5.QtWidgets import QComboBox, QListView

def setup_global_combobox_behavior():
    original_init = QComboBox.__init__

    def custom_init(self, *args, **kwargs):
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

    def custom_show_popup(self):
        original_show_popup(self)

        view = self.view()
        if view is not None:
            text_width = view.sizeHintForColumn(0)
            extra = 6
            target_width = max(self.width(), text_width + extra)
            view.setMinimumWidth(target_width)

    QComboBox.showPopup = custom_show_popup