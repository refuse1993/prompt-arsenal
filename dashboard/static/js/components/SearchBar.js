/**
 * SearchBar Component
 * Reusable search bar with debounce
 */

export class SearchBar {
    constructor(options = {}) {
        this.placeholder = options.placeholder || 'Search...';
        this.onSearch = options.onSearch || null;
        this.debounceMs = options.debounceMs || 300;
        this.debounceTimer = null;
    }

    render() {
        const container = document.createElement('div');
        container.className = 'search-bar';
        container.style.cssText = `
            position: relative;
            width: 100%;
        `;

        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = this.placeholder;
        input.className = 'search-input';
        input.style.cssText = `
            width: 100%;
            padding: var(--spacing-md) var(--spacing-md) var(--spacing-md) 2.5rem;
            background: var(--bg-hover);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: 0.875rem;
            transition: var(--transition);
        `;

        input.addEventListener('focus', () => {
            input.style.borderColor = 'var(--primary)';
        });

        input.addEventListener('blur', () => {
            input.style.borderColor = 'var(--border)';
        });

        input.addEventListener('input', (e) => {
            if (this.onSearch) {
                clearTimeout(this.debounceTimer);
                this.debounceTimer = setTimeout(() => {
                    this.onSearch(e.target.value);
                }, this.debounceMs);
            }
        });

        const icon = document.createElement('div');
        icon.innerHTML = '🔍';
        icon.style.cssText = `
            position: absolute;
            left: var(--spacing-md);
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
            pointer-events: none;
        `;

        container.appendChild(icon);
        container.appendChild(input);

        return container;
    }
}
