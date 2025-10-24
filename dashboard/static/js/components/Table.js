/**
 * Table Component
 * Reusable table component with sorting and pagination
 */

export class Table {
    constructor(options = {}) {
        this.columns = options.columns || [];
        this.data = options.data || [];
        this.sortable = options.sortable !== false;
        this.onRowClick = options.onRowClick || null;
        this.currentSort = { column: null, direction: 'asc' };
    }

    setData(data) {
        this.data = data;
    }

    sort(columnIndex) {
        const column = this.columns[columnIndex];
        if (!column.sortable) return;

        const direction = this.currentSort.column === columnIndex && this.currentSort.direction === 'asc'
            ? 'desc'
            : 'asc';

        this.data.sort((a, b) => {
            const aVal = a[column.key];
            const bVal = b[column.key];

            if (typeof aVal === 'number') {
                return direction === 'asc' ? aVal - bVal : bVal - aVal;
            }

            const aStr = String(aVal).toLowerCase();
            const bStr = String(bVal).toLowerCase();
            return direction === 'asc'
                ? aStr.localeCompare(bStr)
                : bStr.localeCompare(aStr);
        });

        this.currentSort = { column: columnIndex, direction };
    }

    render() {
        const table = document.createElement('table');
        table.className = 'table';

        // Header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');

        this.columns.forEach((column, index) => {
            const th = document.createElement('th');
            th.textContent = column.label;

            if (column.sortable !== false && this.sortable) {
                th.style.cursor = 'pointer';
                th.addEventListener('click', () => {
                    this.sort(index);
                    this.update(table);
                });

                if (this.currentSort.column === index) {
                    th.innerHTML += this.currentSort.direction === 'asc' ? ' ▲' : ' ▼';
                }
            }

            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');

        this.data.forEach(row => {
            const tr = document.createElement('tr');

            if (this.onRowClick) {
                tr.style.cursor = 'pointer';
                tr.addEventListener('click', () => this.onRowClick(row));
            }

            this.columns.forEach(column => {
                const td = document.createElement('td');

                if (column.render) {
                    const rendered = column.render(row[column.key], row);
                    if (typeof rendered === 'string') {
                        td.innerHTML = rendered;
                    } else {
                        td.appendChild(rendered);
                    }
                } else {
                    td.textContent = row[column.key];
                }

                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });

        table.appendChild(tbody);

        return table;
    }

    update(tableElement) {
        const newTable = this.render();
        tableElement.parentNode.replaceChild(newTable, tableElement);
    }
}

// Helper function to create badges
export function createBadge(text, type = 'primary') {
    return `<span class="badge badge-${type}">${text}</span>`;
}

// Helper function to format success/failure
export function formatSuccess(success) {
    return success
        ? createBadge('Success', 'success')
        : createBadge('Failed', 'danger');
}

// Helper function to format dates
export function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}
