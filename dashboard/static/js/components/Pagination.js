/**
 * Pagination Component
 * Reusable pagination component
 */

export class Pagination {
    constructor(options = {}) {
        this.currentPage = options.currentPage || 1;
        this.totalPages = options.totalPages || 1;
        this.onPageChange = options.onPageChange || null;
        this.maxButtons = options.maxButtons || 5;
    }

    setPage(page) {
        if (page < 1 || page > this.totalPages) return;
        this.currentPage = page;
        if (this.onPageChange) {
            this.onPageChange(page);
        }
    }

    setTotalPages(totalPages) {
        this.totalPages = totalPages;
    }

    render() {
        const container = document.createElement('div');
        container.className = 'pagination';
        container.style.cssText = `
            display: flex;
            justify-content: center;
            align-items: center;
            gap: var(--spacing-sm);
            padding: var(--spacing-lg) 0;
        `;

        // Previous button
        const prevBtn = this.createButton('←', () => this.setPage(this.currentPage - 1));
        prevBtn.disabled = this.currentPage === 1;
        container.appendChild(prevBtn);

        // Page buttons
        const buttons = this.getPageButtons();
        buttons.forEach(pageNum => {
            if (pageNum === '...') {
                const ellipsis = document.createElement('span');
                ellipsis.textContent = '...';
                ellipsis.style.cssText = 'padding: 0 var(--spacing-sm); color: var(--text-secondary);';
                container.appendChild(ellipsis);
            } else {
                const btn = this.createButton(
                    pageNum,
                    () => this.setPage(pageNum),
                    pageNum === this.currentPage
                );
                container.appendChild(btn);
            }
        });

        // Next button
        const nextBtn = this.createButton('→', () => this.setPage(this.currentPage + 1));
        nextBtn.disabled = this.currentPage === this.totalPages;
        container.appendChild(nextBtn);

        return container;
    }

    createButton(text, onClick, isActive = false) {
        const button = document.createElement('button');
        button.textContent = text;
        button.className = 'pagination-btn';

        const baseStyle = `
            padding: var(--spacing-sm) var(--spacing-md);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            background: ${isActive ? 'var(--primary)' : 'var(--bg-card)'};
            color: ${isActive ? 'white' : 'var(--text-primary)'};
            cursor: pointer;
            font-size: 0.875rem;
            transition: var(--transition);
            min-width: 40px;
        `;

        button.style.cssText = baseStyle;

        button.addEventListener('click', onClick);

        button.addEventListener('mouseenter', () => {
            if (!isActive && !button.disabled) {
                button.style.background = 'var(--bg-hover)';
            }
        });

        button.addEventListener('mouseleave', () => {
            if (!isActive) {
                button.style.background = 'var(--bg-card)';
            }
        });

        if (button.disabled) {
            button.style.opacity = '0.5';
            button.style.cursor = 'not-allowed';
        }

        return button;
    }

    getPageButtons() {
        const buttons = [];
        const { currentPage, totalPages, maxButtons } = this;

        if (totalPages <= maxButtons) {
            for (let i = 1; i <= totalPages; i++) {
                buttons.push(i);
            }
        } else {
            const leftSide = Math.floor((maxButtons - 3) / 2);
            const rightSide = Math.ceil((maxButtons - 3) / 2);

            let start = currentPage - leftSide;
            let end = currentPage + rightSide;

            if (start < 2) {
                start = 2;
                end = maxButtons - 1;
            }

            if (end > totalPages - 1) {
                end = totalPages - 1;
                start = totalPages - maxButtons + 2;
            }

            buttons.push(1);

            if (start > 2) {
                buttons.push('...');
            }

            for (let i = start; i <= end; i++) {
                buttons.push(i);
            }

            if (end < totalPages - 1) {
                buttons.push('...');
            }

            if (totalPages > 1) {
                buttons.push(totalPages);
            }
        }

        return buttons;
    }

    update(element) {
        const newPagination = this.render();
        element.parentNode.replaceChild(newPagination, element);
        return newPagination;
    }
}
