import pandas as pd
from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QVariant
)
from PyQt5.QtGui import QBrush, QColor
# Lightweight Qt model exposing a pandas.DataFrame to QTableView
#This follows the MVC design pattern: model, view, controller.
#So this class works as the model for the view in the table.
#That means, here, inside this class lives the data shown in the table.
#remember that an abstract class can't be instanciated directly
#because it doesn't have the methods with the tag
#@abstractmethod implemented. Just defines
#some sort of contract in which the subclass of the abstractclass must
#implement those methods to be instanciated. Those methods are the ones
#we are doing polymorphism here: rowCount,columnCount,data
class PandasModel(QAbstractTableModel):
    """Lightweight Qt model for displaying pandas DataFrames in tables.

    This class implements the MVC model pattern, exposing a pandas
    DataFrame to a QTableView. Data is loaded lazily without creating
    per-cell QTableWidgetItem objects. Supports column highlighting
    for columns containing NaN values and sorting.

    Attributes
    ----------
    _df : pd.DataFrame
        The underlying pandas DataFrame containing table data.
    highlight_cols : set
        Set of column names to highlight in the table view.
    highlight_color : QColor
        Color used for highlighting columns with missing values.

    Methods
    -------
    rowCount(parent)
        Return the number of rows in the DataFrame.
    columnCount(parent)
        Return the number of columns in the DataFrame.
    data(index, role)
        Return data at a given index for display or editing.
    headerData(section, orientation, role)
        Return header labels and formatting.
    set_highlight_by_missing(columns)
        Highlight columns containing NaN values.
    sort(column, order)
        Sort the DataFrame by a column.
    set_dataframe(df)
        Replace the DataFrame and refresh the view.
    """

    def __init__(self, df: pd.DataFrame, parent=None) -> None:
        """Initialize the PandasModel with a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to display in the table.
        parent : QObject, optional
            Parent Qt object. Default is None.

        Returns
        -------
        None
        """
        super().__init__(parent)
        self._df = df
        self.highlight_cols = set() # Here we are going to store
        #our highlighted columns, a set to avoid duplicates.
        self.highlight_color = QColor("#290908")  # just the color

    def rowCount(self, parent=None) -> int:
        """Return the number of rows in the DataFrame.

        Parameters
        ----------
        parent : QModelIndex, optional
            Parent model index. Default is None.

        Returns
        -------
        int
            Number of rows in the underlying DataFrame.
        """
        if parent and parent.isValid():
            return 0
        return len(self._df.index)

    def columnCount(self, parent=None) -> int:
        """Return the number of columns in the DataFrame.

        Parameters
        ----------
        parent : QModelIndex, optional
            Parent model index. Default is None.

        Returns
        -------
        int
            Number of columns in the underlying DataFrame.
        """
        if parent and parent.isValid():
            return 0
        return len(self._df.columns)

    def data(
            self,
            index,
            role=Qt.DisplayRole
    ) -> QVariant:
        """Return data at a given index for display or editing.

        Parameters
        ----------
        index : QModelIndex
            Index of the cell to retrieve.
        role : Qt.ItemDataRole, optional
            Role determining what data to return (Display, Edit,
            Background, etc.). Default is Qt.DisplayRole.

        Returns
        -------
        QVariant
            Data converted to string for display, or QBrush for
            background highlighting in marked columns. Returns
            QVariant() if index is invalid.
        """
        if not index.isValid():
            return QVariant()

        row = index.row()
        col = index.column()
        val = self._df.iat[row, col]
        col_name = self._df.columns[col]

        if role in (Qt.DisplayRole, Qt.EditRole):
            return "" if pd.isna(val) else str(val)

        if role == Qt.BackgroundRole:
            # If the column is marked, we paint the ENTIRE column.
            if col_name in self.highlight_cols:
                return QBrush(self.highlight_color)

        return QVariant()

    def headerData(
            self,
            section: int,
            orientation,
            role=Qt.DisplayRole
    ) -> QVariant:
        """Return header label and formatting for columns/rows.

        Parameters
        ----------
        section : int
            Column or row index.
        orientation : Qt.Orientation
            Qt.Horizontal for column headers, Qt.Vertical for rows.
        role : Qt.ItemDataRole, optional
            Role determining what to return (DisplayRole,
            BackgroundRole). Default is Qt.DisplayRole.

        Returns
        -------
        QVariant
            Header text or background brush for highlighted columns.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(section)

        if role == Qt.BackgroundRole and orientation == Qt.Horizontal:
            col_name = self._df.columns[section]
            if col_name in self.highlight_cols:
                return QBrush(self.highlight_color)

        return QVariant()

    def set_highlight_by_missing(self, columns: list) -> bool:
        """Highlight columns that contain at least one NaN value.

        Identifies columns with missing values and marks them for
        visual highlighting in both cells and headers. Updates the view.

        Parameters
        ----------
        columns : list
            Column names to check for missing values.

        Returns
        -------
        bool
            True if any columns were marked for highlighting, False
            otherwise.
        """
        columns = columns or []
        # Looks complicated at first glance but it's not a big deal
        # First of all is a set that avoid selecting duplicates
        # columns(it could happen when you select the same column in
        # input-output, it's a trivial regression but you never know
        # what the user does). This is a set comprehension.
        # first we iterate over columns. Each c is a column
        # then in self._df[c] we select the column, .isna()
        # returns a dataframe replacing each element of the
        # column(not in-place obviusly not replacing the elements of the
        # original column of the dataframe) with boolean values,
        # True if it's nan, False otherwise. .any() returns
        # a boolean if the new pandas has at least one True,
        # that means, if has at least one NaN element in that column.
        # So we add c, that column to
        # the set.
        self.highlight_cols = {
            c for c in columns if
            c in self._df.columns
            and self._df[c].isna().any()
        }

        # Repintar celdas y encabezados
        if self.rowCount() and self.columnCount():
            top_left = self.index(0, 0)
            bottom_right = self.index(self.rowCount() - 1,
                                      self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.BackgroundRole])
            self.headerDataChanged.emit(Qt.Horizontal, 0,
                                        self.columnCount() - 1)
        if self.highlight_cols:
            return True
        return False

    def sort(self, column: int, order) -> None:
        """Sort the DataFrame by a column when header is clicked.

        Uses a stable sort to maintain relative order of tied values,
        improving UX when toggling sort direction. Resets the index
        after sorting and emits layout change signals.

        Parameters
        ----------
        column : int
            Column index to sort by.
        order : Qt.SortOrder
            Qt.AscendingOrder or Qt.DescendingOrder.

        Returns
        -------
        None
        """
        self.layoutAboutToBeChanged.emit()
        ascending = (order == Qt.AscendingOrder)
        # Stable sort keeps relative order of ties
        # (nicer UX when toggling)
        self._df.sort_values(
            by=self._df.columns[column],
            ascending=ascending,
            inplace=True,
            kind="mergesort"
        )
        self._df.reset_index(drop=True, inplace=True)
        self.layoutChanged.emit()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Replace the DataFrame and refresh the table view.

        Updates the internal DataFrame and signals the view to reset
        and repaint all cells.

        Parameters
        ----------
        df : pd.DataFrame
            New DataFrame to display in the table.

        Returns
        -------
        None
        """
        self.beginResetModel()
        self._df = df
        self.endResetModel()