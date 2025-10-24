/**
 * Card Component
 * Reusable card component for dashboard
 */

export class Card {
    constructor(options = {}) {
        this.title = options.title || '';
        this.icon = options.icon || '';
        this.className = options.className || '';
        this.onClick = options.onClick || null;
    }

    render(content) {
        const card = document.createElement('div');
        card.className = `card ${this.className}`;

        if (this.onClick) {
            card.style.cursor = 'pointer';
            card.addEventListener('click', this.onClick);
        }

        const header = this.title || this.icon ? `
            <div class="card-header">
                <h3 class="card-title">${this.icon} ${this.title}</h3>
            </div>
        ` : '';

        card.innerHTML = `
            ${header}
            <div class="card-body">
                ${content}
            </div>
        `;

        return card;
    }

    static createStatCard(value, label, icon = '') {
        const card = document.createElement('div');
        card.className = 'card stat-card';
        card.innerHTML = `
            <div class="stat-value">${icon} ${value.toLocaleString()}</div>
            <div class="stat-label">${label}</div>
        `;
        return card;
    }

    static createInfoCard(title, items, icon = '') {
        const card = document.createElement('div');
        card.className = 'card';

        const itemsHTML = items.map(item => `
            <div class="flex-between" style="padding: 0.5rem 0; border-bottom: 1px solid var(--border);">
                <span style="color: var(--text-secondary);">${item.label}</span>
                <span style="font-weight: 600;">${item.value}</span>
            </div>
        `).join('');

        card.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">${icon} ${title}</h3>
            </div>
            <div class="card-body">
                ${itemsHTML}
            </div>
        `;

        return card;
    }
}
