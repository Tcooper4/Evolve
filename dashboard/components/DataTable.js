import React, { useState, useMemo } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Paper,
  TextField,
  Box,
} from '@mui/material';

const DataTable = ({
  columns,
  data,
  defaultSort = { field: '', direction: 'asc' },
  pageSize = 10,
  theme = 'light',
}) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(pageSize);
  const [sortConfig, setSortConfig] = useState(defaultSort);
  const [filters, setFilters] = useState({});

  const handleSort = (field) => {
    setSortConfig({
      field,
      direction:
        sortConfig.field === field && sortConfig.direction === 'asc'
          ? 'desc'
          : 'asc',
    });
  };

  const handleFilterChange = (field, value) => {
    setFilters((prev) => ({
      ...prev,
      [field]: value,
    }));
    setPage(0);
  };

  const filteredData = useMemo(() => {
    return data.filter((row) =>
      Object.entries(filters).every(
        ([field, value]) =>
          !value ||
          String(row[field])
            .toLowerCase()
            .includes(String(value).toLowerCase())
      )
    );
  }, [data, filters]);

  const sortedData = useMemo(() => {
    if (!sortConfig.field) return filteredData;

    return [...filteredData].sort((a, b) => {
      const aValue = a[sortConfig.field];
      const bValue = b[sortConfig.field];

      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }, [filteredData, sortConfig]);

  const paginatedData = useMemo(() => {
    const start = page * rowsPerPage;
    return sortedData.slice(start, start + rowsPerPage);
  }, [sortedData, page, rowsPerPage]);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const tableStyles = {
    backgroundColor: theme === 'dark' ? '#1E1E1E' : '#FFFFFF',
    color: theme === 'dark' ? '#FFFFFF' : '#000000',
  };

  const cellStyles = {
    color: theme === 'dark' ? '#FFFFFF' : '#000000',
    borderBottom: `1px solid ${
      theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
    }`,
  };

  return (
    <Paper
      sx={{
        width: '100%',
        overflow: 'hidden',
        backgroundColor: tableStyles.backgroundColor,
      }}
    >
      <Box sx={{ p: 2 }}>
        {columns.map((column) => (
          <TextField
            key={column.field}
            label={`Filter ${column.headerName}`}
            variant="outlined"
            size="small"
            value={filters[column.field] || ''}
            onChange={(e) => handleFilterChange(column.field, e.target.value)}
            sx={{ mr: 2, mb: 2 }}
          />
        ))}
      </Box>
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell
                  key={column.field}
                  style={cellStyles}
                  sortDirection={
                    sortConfig.field === column.field
                      ? sortConfig.direction
                      : false
                  }
                >
                  <TableSortLabel
                    active={sortConfig.field === column.field}
                    direction={
                      sortConfig.field === column.field
                        ? sortConfig.direction
                        : 'asc'
                    }
                    onClick={() => handleSort(column.field)}
                    sx={{ color: cellStyles.color }}
                  >
                    {column.headerName}
                  </TableSortLabel>
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {paginatedData.map((row, index) => (
              <TableRow key={index}>
                {columns.map((column) => (
                  <TableCell key={column.field} style={cellStyles}>
                    {row[column.field]}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[5, 10, 25, 50]}
        component="div"
        count={filteredData.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
        sx={{
          color: cellStyles.color,
          '.MuiTablePagination-select': {
            color: cellStyles.color,
          },
          '.MuiTablePagination-selectIcon': {
            color: cellStyles.color,
          },
        }}
      />
    </Paper>
  );
};

export default DataTable; 