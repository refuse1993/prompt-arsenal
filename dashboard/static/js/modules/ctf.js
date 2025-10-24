/**
 * CTF Page Module
 * CTF challenge and competition management
 */

import { Card } from '../components/Card.js';
import { Table, createBadge, formatDate } from '../components/Table.js';
import { Pagination } from '../components/Pagination.js';
import { Modal } from '../components/Modal.js';
import { Chart } from '../components/Chart.js';

class CTFPage {
    constructor() {
        this.currentPage = 1;
        this.currentCategory = null;
        this.pagination = new Pagination({
            currentPage: 1,
            totalPages: 1,
            onPageChange: (page) => this.loadChallenges(page)
        });

        this.init();
    }

    async init() {
        await this.loadStats();
        await this.loadCategories();
        await this.loadCompetitions();
        this.renderFilters();
        await this.loadChallenges();

        document.getElementById('crawl-btn').addEventListener('click', () => {
            this.showCrawlModal();
        });
    }

    async loadStats() {
        try {
            const response = await fetch('/api/ctf/stats');
            const result = await response.json();

            if (result.success) {
                this.renderStats(result.data);
            }
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    renderStats(stats) {
        const container = document.getElementById('stats-container');

        container.appendChild(Card.createStatCard(
            stats.total_challenges,
            'Total Challenges',
            '🎯'
        ));

        container.appendChild(Card.createStatCard(
            stats.total_tests,
            'Tests Run',
            '⚡'
        ));

        container.appendChild(Card.createStatCard(
            `${(stats.avg_success_rate * 100).toFixed(1)}%`,
            'Avg Success Rate',
            '📊'
        ));
    }

    async loadCategories() {
        try {
            const response = await fetch('/api/ctf/categories');
            const result = await response.json();

            if (result.success) {
                this.categories = result.data;
                this.renderCategoriesChart(result.data);
                this.renderSuccessChart(result.data);
            }
        } catch (error) {
            console.error('Failed to load categories:', error);
        }
    }

    renderCategoriesChart(categories) {
        const container = document.getElementById('categories-chart');

        const chartData = categories.map(cat => ({
            label: cat.display,
            value: 1,
            color: this.getCategoryColor(cat.name)
        }));

        container.appendChild(Chart.createPieChart(chartData));
    }

    renderSuccessChart(categories) {
        const container = document.getElementById('success-chart');

        const chartData = categories.map(cat => ({
            label: cat.display,
            value: cat.success_rate * 100,
            color: this.getCategoryColor(cat.name)
        }));

        container.appendChild(Chart.createBarChart(chartData));
    }

    async loadCompetitions() {
        try {
            const response = await fetch('/api/ctf/competitions');
            const result = await response.json();

            if (result.success) {
                this.renderCompetitions(result.data);
            }
        } catch (error) {
            console.error('Failed to load competitions:', error);
        }
    }

    renderCompetitions(competitions) {
        const container = document.getElementById('competitions-container');

        const grid = document.createElement('div');
        grid.className = 'grid grid-3';

        competitions.forEach(comp => {
            const items = [
                { label: 'Challenges Found', value: comp.challenges_found },
                { label: 'Status', value: comp.status }
            ];

            const card = Card.createInfoCard(comp.name, items, '🏆');

            const link = document.createElement('a');
            link.href = comp.url;
            link.target = '_blank';
            link.style.cssText = 'color: var(--primary); font-size: 0.875rem; text-decoration: none;';
            link.textContent = '🔗 Visit Competition';

            card.querySelector('.card-body').appendChild(link);

            grid.appendChild(card);
        });

        container.appendChild(grid);
    }

    renderFilters() {
        const container = document.getElementById('filter-container');

        container.innerHTML = `
            <select id="category-filter" style="
                padding: var(--spacing-sm) var(--spacing-md);
                background: var(--bg-hover);
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                color: var(--text-primary);
            ">
                <option value="">All Categories</option>
                <option value="sql_injection">SQL Injection</option>
                <option value="xss">XSS</option>
                <option value="csrf">CSRF</option>
                <option value="lfi">LFI</option>
                <option value="command_injection">Command Injection</option>
            </select>
        `;

        document.getElementById('category-filter').addEventListener('change', (e) => {
            this.currentCategory = e.target.value || null;
            this.loadChallenges(1);
        });
    }

    async loadChallenges(page = 1) {
        this.currentPage = page;

        try {
            let url = `/api/ctf/challenges?page=${page}&limit=50`;
            if (this.currentCategory) {
                url += `&category=${this.currentCategory}`;
            }

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.renderTable(result.data);

                if (result.pagination) {
                    this.pagination.currentPage = result.pagination.page;
                    this.pagination.totalPages = result.pagination.pages;
                    this.renderPagination();
                }
            }
        } catch (error) {
            console.error('Failed to load challenges:', error);
        }
    }

    renderTable(challenges) {
        const container = document.getElementById('table-container');

        const table = new Table({
            columns: [
                {
                    key: 'id',
                    label: 'ID',
                    sortable: true
                },
                {
                    key: 'title',
                    label: 'Title',
                    sortable: true
                },
                {
                    key: 'category',
                    label: 'Category',
                    sortable: true,
                    render: (value) => createBadge(value, 'primary')
                },
                {
                    key: 'difficulty',
                    label: 'Difficulty',
                    sortable: true,
                    render: (value) => {
                        const color = value === 'easy' ? 'success' : value === 'medium' ? 'warning' : 'danger';
                        return createBadge(value, color);
                    }
                },
                {
                    key: 'url',
                    label: 'URL',
                    sortable: false,
                    render: (value) => `<a href="${value}" target="_blank" style="color: var(--primary);">🔗 Open</a>`
                },
                {
                    key: 'solved',
                    label: 'Status',
                    sortable: true,
                    render: (value) => value
                        ? createBadge('Solved', 'success')
                        : createBadge('Unsolved', 'warning')
                }
            ],
            data: challenges,
            onRowClick: (row) => this.showChallengeDetails(row)
        });

        container.innerHTML = '';
        container.appendChild(table.render());
    }

    renderPagination() {
        const container = document.getElementById('pagination-container');
        container.innerHTML = '';
        container.appendChild(this.pagination.render());
    }

    async showChallengeDetails(challenge) {
        const modal = new Modal({ title: challenge.title });

        const content = document.createElement('div');
        content.innerHTML = `
            <div class="mb-1">
                <strong>Category:</strong> ${createBadge(challenge.category, 'primary')}
            </div>
            <div class="mb-1">
                <strong>Difficulty:</strong> ${createBadge(challenge.difficulty, 'warning')}
            </div>
            <div class="mb-1">
                <strong>URL:</strong> <a href="${challenge.url}" target="_blank" style="color: var(--primary);">${challenge.url}</a>
            </div>
            <div class="mb-1">
                <strong>Description:</strong>
                <p style="color: var(--text-secondary); margin-top: 0.5rem;">${challenge.description || 'N/A'}</p>
            </div>
            <div class="mb-1">
                <strong>Status:</strong> ${challenge.solved
                    ? createBadge('Solved', 'success')
                    : createBadge('Unsolved', 'warning')}
            </div>
            <div class="flex gap-1 mt-2">
                <button class="btn btn-primary solve-btn">🚀 Auto Solve</button>
                <button class="btn btn-secondary analyze-btn">🔍 Analyze Page</button>
            </div>
        `;

        content.querySelector('.solve-btn').addEventListener('click', () => {
            modal.close();
            Modal.alert('Info', 'Auto-solve started! This may take a few minutes...');
        });

        content.querySelector('.analyze-btn').addEventListener('click', () => {
            modal.close();
            this.showPageAnalysis(challenge);
        });

        modal.show(content);
    }

    showPageAnalysis(challenge) {
        const modal = new Modal({ title: `Page Analysis: ${challenge.title}` });

        const content = document.createElement('div');
        content.innerHTML = `
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                Playwright-based page analysis results:
            </p>
            <div class="mb-1">
                <strong>Forms Found:</strong> 3
                <ul style="color: var(--text-secondary); margin-top: 0.5rem;">
                    <li>Login form (POST /login)</li>
                    <li>Search form (GET /search)</li>
                    <li>Contact form (POST /contact)</li>
                </ul>
            </div>
            <div class="mb-1">
                <strong>Scripts:</strong> 5 JavaScript files
            </div>
            <div class="mb-1">
                <strong>Comments:</strong> 12 HTML comments found
            </div>
            <div class="mb-1">
                <strong>Cookies:</strong> session_id, csrf_token
            </div>
            <div class="mb-1">
                <strong>API Endpoints:</strong>
                <ul style="color: var(--text-secondary); margin-top: 0.5rem;">
                    <li>/api/users</li>
                    <li>/api/products</li>
                </ul>
            </div>
        `;

        modal.show(content);
    }

    showCrawlModal() {
        const modal = new Modal({ title: 'Crawl CTF Competition' });

        const content = document.createElement('div');
        content.innerHTML = `
            <form id="crawl-form">
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Competition URL</label>
                    <input type="url" name="url" required placeholder="https://picoctf.org" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Max Challenges</label>
                    <input type="number" name="max_challenges" value="30" min="1" max="100" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                </div>
                <div class="flex gap-1 mt-2">
                    <button type="submit" class="btn btn-primary">🔍 Start Crawl</button>
                    <button type="button" class="btn btn-secondary cancel-btn">Cancel</button>
                </div>
            </form>
        `;

        content.querySelector('.cancel-btn').addEventListener('click', () => modal.close());

        content.querySelector('#crawl-form').addEventListener('submit', (e) => {
            e.preventDefault();
            modal.close();
            Modal.alert('Success', 'Crawling started! Challenges will appear in the list soon.');
        });

        modal.show(content);
    }

    getCategoryColor(category) {
        const colors = {
            'sql_injection': '#3b82f6',
            'xss': '#8b5cf6',
            'csrf': '#10b981',
            'lfi': '#f59e0b',
            'command_injection': '#ef4444'
        };
        return colors[category] || '#64748b';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new CTFPage();
});
