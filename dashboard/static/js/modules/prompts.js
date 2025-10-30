/**
 * Prompts Page Module
 * Text prompt management
 */

import { Card } from '../components/Card.js';
import { Table, createBadge, formatDate } from '../components/Table.js';
import { SearchBar } from '../components/SearchBar.js';
import { Pagination } from '../components/Pagination.js';
import { Modal } from '../components/Modal.js';

class PromptsPage {
    constructor() {
        this.currentPage = 1;
        this.currentCategory = null;
        this.currentModel = null;
        this.searchQuery = '';

        // Classification filters
        this.currentPurpose = null;
        this.currentRiskCategory = null;
        this.currentTechnique = null;
        this.currentModality = null;

        this.pagination = new Pagination({
            currentPage: 1,
            totalPages: 1,
            onPageChange: (page) => this.loadPrompts(page)
        });

        this.init();
    }

    async init() {
        await this.loadCategories();
        await this.renderFilters();
        this.renderSearchBar();
        await this.loadPrompts();
    }

    async loadCategories() {
        try {
            const response = await fetch('/api/prompts/categories');
            const result = await response.json();

            if (result.success) {
                this.renderCategories(result.data);
            }
        } catch (error) {
            console.error('Failed to load categories:', error);
        }
    }

    renderCategories(categories) {
        const container = document.getElementById('categories-container');
        container.style.cssText = `
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        `;

        // All button
        const allBtn = this.createFilterButton('All', null, categories.reduce((sum, c) => sum + c.count, 0));
        container.appendChild(allBtn);

        // Category buttons
        categories.forEach(category => {
            const btn = this.createFilterButton(
                category.name,
                category.name,
                category.count
            );
            container.appendChild(btn);
        });
    }

    createFilterButton(label, category, count) {
        const btn = document.createElement('button');
        const isActive = this.currentCategory === category;

        btn.style.cssText = `
            display: inline-flex;
            align-items: center;
            gap: 0.625rem;
            padding: 0.625rem 1.25rem;
            font-size: 0.9375rem;
            border-radius: var(--radius-md);
            border: 1px solid ${isActive ? 'hsl(var(--primary))' : 'hsl(var(--border))'};
            background: ${isActive ? 'hsl(var(--primary))' : 'hsl(var(--card))'};
            color: ${isActive ? 'hsl(var(--primary-foreground))' : 'hsl(var(--foreground))'};
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
            box-shadow: ${isActive ? '0 2px 4px rgba(0,0,0,0.1)' : 'none'};
        `;

        btn.innerHTML = `
            <span style="font-weight: 600;">${label}</span>
            <span style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 2rem;
                height: 1.5rem;
                padding: 0 0.5rem;
                background: ${isActive ? 'hsl(var(--primary-foreground) / 0.2)' : 'hsl(var(--muted) / 0.5)'};
                border-radius: var(--radius-sm);
                font-size: 0.8125rem;
                font-weight: 700;
            ">${count}</span>
        `;

        btn.addEventListener('mouseenter', () => {
            if (!isActive) {
                btn.style.background = 'hsl(var(--accent))';
                btn.style.borderColor = 'hsl(var(--primary) / 0.5)';
            }
        });

        btn.addEventListener('mouseleave', () => {
            if (!isActive) {
                btn.style.background = 'hsl(var(--background))';
                btn.style.borderColor = 'hsl(var(--border))';
            }
        });

        btn.addEventListener('click', () => {
            this.currentCategory = category;
            this.loadPrompts(1);
        });

        return btn;
    }

    async renderFilters() {
        const container = document.getElementById('filter-container');

        // Fetch classification options
        let classificationOptions = null;
        try {
            const response = await fetch('/api/prompts/classification-options');
            const result = await response.json();
            if (result.success) {
                classificationOptions = result.data;
            }
        } catch (error) {
            console.error('Failed to load classification options:', error);
        }

        // Fetch models
        let models = null;
        try {
            const response = await fetch('/api/prompts/models');
            const result = await response.json();
            if (result.success && result.data.length > 0) {
                models = result.data;
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }

        // Container for all filters
        const filterGrid = document.createElement('div');
        filterGrid.style.cssText = `
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.75rem;
            padding: 1rem;
            background: hsl(var(--card));
            border: 1px solid hsl(var(--border));
            border-radius: var(--radius-md);
            margin-bottom: 1rem;
        `;

        // Purpose filter
        if (classificationOptions?.purpose?.length > 0) {
            filterGrid.appendChild(this.createFilterDropdown(
                '🎯 Purpose',
                'purpose',
                classificationOptions.purpose.map(p => ({
                    value: p.purpose,
                    label: p.purpose === 'offensive' ? '⚔️ Offensive' : '🛡️ Defensive',
                    count: p.count
                })),
                (value) => {
                    this.currentPurpose = value;
                    this.loadPrompts(1);
                }
            ));
        }

        // Risk Category filter
        if (classificationOptions?.risk_category?.length > 0) {
            const riskIcons = {
                'security': '🔒',
                'safety': '⚠️',
                'ethics': '🎭',
                'compliance': '📋',
                'misinformation': '📰'
            };
            filterGrid.appendChild(this.createFilterDropdown(
                '⚠️ Risk Category',
                'risk_category',
                classificationOptions.risk_category.map(r => ({
                    value: r.risk_category,
                    label: `${riskIcons[r.risk_category] || ''} ${r.risk_category}`,
                    count: r.count
                })),
                (value) => {
                    this.currentRiskCategory = value;
                    this.loadPrompts(1);
                }
            ));
        }

        // Technique filter
        if (classificationOptions?.technique?.length > 0) {
            filterGrid.appendChild(this.createFilterDropdown(
                '🔧 Technique',
                'technique',
                classificationOptions.technique.map(t => ({
                    value: t.technique,
                    label: t.technique.replace(/_/g, ' '),
                    count: t.count
                })),
                (value) => {
                    this.currentTechnique = value;
                    this.loadPrompts(1);
                }
            ));
        }

        // Modality filter
        if (classificationOptions?.modality?.length > 0) {
            filterGrid.appendChild(this.createFilterDropdown(
                '🎨 Modality',
                'modality',
                classificationOptions.modality.map(m => ({
                    value: m.modality,
                    label: m.modality.replace(/_/g, ' '),
                    count: m.count
                })),
                (value) => {
                    this.currentModality = value;
                    this.loadPrompts(1);
                }
            ));
        }

        // Model filter
        if (models) {
            filterGrid.appendChild(this.createFilterDropdown(
                '🤖 Model',
                'model',
                models.map(m => ({
                    value: `${m.provider}/${m.model}`,
                    label: `${m.provider}/${m.model}`,
                    count: m.count
                })),
                (value) => {
                    this.currentModel = value;
                    this.loadPrompts(1);
                }
            ));
        }

        container.appendChild(filterGrid);
    }

    createFilterDropdown(label, name, options, onChange) {
        const wrapper = document.createElement('div');

        const labelEl = document.createElement('label');
        labelEl.textContent = label;
        labelEl.style.cssText = `
            display: block;
            font-weight: 600;
            font-size: 0.8125rem;
            color: hsl(var(--foreground));
            margin-bottom: 0.375rem;
        `;

        const select = document.createElement('select');
        select.style.cssText = `
            width: 100%;
            padding: 0.5rem 0.75rem;
            background: hsl(var(--background));
            border: 1px solid hsl(var(--border));
            border-radius: var(--radius-md);
            color: hsl(var(--foreground));
            font-size: 0.875rem;
            cursor: pointer;
        `;

        // Add "All" option
        const allOption = document.createElement('option');
        allOption.value = '';
        allOption.textContent = `All (${options.reduce((sum, o) => sum + o.count, 0)})`;
        select.appendChild(allOption);

        // Add options
        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = `${opt.label} (${opt.count})`;
            select.appendChild(option);
        });

        select.addEventListener('change', (e) => {
            onChange(e.target.value || null);
        });

        wrapper.appendChild(labelEl);
        wrapper.appendChild(select);
        return wrapper;
    }

    renderSearchBar() {
        const container = document.getElementById('search-container');
        const searchBar = new SearchBar({
            placeholder: 'Search prompts...',
            onSearch: (query) => {
                this.searchQuery = query;
                this.loadPrompts(1);
            }
        });
        container.appendChild(searchBar.render());
    }

    async loadPrompts(page = 1) {
        this.currentPage = page;

        try {
            let url = `/api/prompts/list?page=${page}&limit=50`;
            if (this.currentCategory) {
                url += `&category=${encodeURIComponent(this.currentCategory)}`;
            }
            if (this.currentModel) {
                url += `&model=${encodeURIComponent(this.currentModel)}`;
            }

            // Add classification filters
            if (this.currentPurpose) {
                url += `&purpose=${encodeURIComponent(this.currentPurpose)}`;
            }
            if (this.currentRiskCategory) {
                url += `&risk_category=${encodeURIComponent(this.currentRiskCategory)}`;
            }
            if (this.currentTechnique) {
                url += `&technique=${encodeURIComponent(this.currentTechnique)}`;
            }
            if (this.currentModality) {
                url += `&modality=${encodeURIComponent(this.currentModality)}`;
            }

            if (this.searchQuery) {
                url = `/api/prompts/search?q=${encodeURIComponent(this.searchQuery)}`;
                if (this.currentCategory) {
                    url += `&category=${encodeURIComponent(this.currentCategory)}`;
                }
                if (this.currentModel) {
                    url += `&model=${encodeURIComponent(this.currentModel)}`;
                }
                // Add classification filters to search
                if (this.currentPurpose) {
                    url += `&purpose=${encodeURIComponent(this.currentPurpose)}`;
                }
                if (this.currentRiskCategory) {
                    url += `&risk_category=${encodeURIComponent(this.currentRiskCategory)}`;
                }
                if (this.currentTechnique) {
                    url += `&technique=${encodeURIComponent(this.currentTechnique)}`;
                }
                if (this.currentModality) {
                    url += `&modality=${encodeURIComponent(this.currentModality)}`;
                }
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
            console.error('Failed to load prompts:', error);
        }
    }

    renderTable(prompts) {
        const container = document.getElementById('table-container');

        const table = new Table({
            columns: [
                {
                    key: 'id',
                    label: 'ID',
                    sortable: true,
                    render: (value) => `<span style="font-weight: 600; color: hsl(var(--muted-foreground));">#${value}</span>`
                },
                {
                    key: 'category',
                    label: 'Category',
                    sortable: true,
                    render: (value) => createBadge(value, 'primary')
                },
                {
                    key: 'classification',
                    label: 'Classification',
                    sortable: false,
                    render: (value, row) => {
                        const badges = [];

                        if (row.purpose) {
                            const icon = row.purpose === 'offensive' ? '⚔️' : '🛡️';
                            const color = row.purpose === 'offensive' ? 'hsl(var(--destructive))' : 'hsl(var(--success))';
                            badges.push(`<span style="
                                display: inline-flex;
                                align-items: center;
                                gap: 0.25rem;
                                padding: 0.125rem 0.5rem;
                                background: ${color}33;
                                color: ${color};
                                border-radius: var(--radius-sm);
                                font-size: 0.75rem;
                                font-weight: 600;
                            ">${icon} ${row.purpose}</span>`);
                        }

                        if (row.risk_category) {
                            const riskIcons = {
                                'security': '🔒',
                                'safety': '⚠️',
                                'ethics': '🎭',
                                'compliance': '📋',
                                'misinformation': '📰'
                            };
                            const icon = riskIcons[row.risk_category] || '⚠️';
                            badges.push(`<span style="
                                display: inline-flex;
                                align-items: center;
                                gap: 0.25rem;
                                padding: 0.125rem 0.5rem;
                                background: hsl(var(--muted) / 0.3);
                                color: hsl(var(--foreground));
                                border-radius: var(--radius-sm);
                                font-size: 0.75rem;
                            ">${icon} ${row.risk_category}</span>`);
                        }

                        return badges.length > 0
                            ? `<div style="display: flex; gap: 0.25rem; flex-wrap: wrap;">${badges.join('')}</div>`
                            : '<span style="color: hsl(var(--muted-foreground)); font-size: 0.75rem;">-</span>';
                    }
                },
                {
                    key: 'payload',
                    label: 'Payload',
                    sortable: false,
                    render: (value) => {
                        const truncated = value.length > 60 ? value.substring(0, 60) + '...' : value;
                        return `<span style="font-family: monospace; font-size: 0.875rem; color: hsl(var(--foreground));">${truncated}</span>`;
                    }
                },
                {
                    key: 'success_rate',
                    label: 'Success Rate',
                    sortable: true,
                    render: (value, row) => {
                        if (row.test_count === 0) {
                            return '<span style="color: hsl(var(--muted-foreground)); font-size: 0.875rem;">No tests</span>';
                        }
                        const rate = Math.round(value);
                        const color = rate >= 70 ? 'hsl(var(--success))' : rate >= 40 ? 'hsl(142.1 70.6% 45.3%)' : 'hsl(var(--muted-foreground))';
                        return `
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <div style="
                                    flex: 1;
                                    height: 6px;
                                    background: hsl(var(--muted) / 0.3);
                                    border-radius: 3px;
                                    overflow: hidden;
                                    max-width: 80px;
                                ">
                                    <div style="
                                        height: 100%;
                                        width: ${rate}%;
                                        background: ${color};
                                        transition: width 0.3s;
                                    "></div>
                                </div>
                                <span style="font-weight: 600; color: ${color}; min-width: 3rem; font-size: 0.875rem;">${rate}%</span>
                            </div>
                        `;
                    }
                },
                {
                    key: 'test_count',
                    label: 'Tests',
                    sortable: true,
                    render: (value, row) => {
                        if (value === 0) {
                            return '<span style="color: hsl(var(--muted-foreground)); font-size: 0.875rem;">0</span>';
                        }
                        return `
                            <div style="display: flex; align-items: center; gap: 0.25rem;">
                                <span style="font-weight: 600; color: hsl(var(--foreground));">${value}</span>
                                <span style="font-size: 0.75rem; color: hsl(var(--muted-foreground));">
                                    (${row.success_count}✓)
                                </span>
                            </div>
                        `;
                    }
                },
                {
                    key: 'created_at',
                    label: 'Created',
                    sortable: true,
                    render: formatDate
                }
            ],
            data: prompts,
            onRowClick: (row) => this.showPromptDetails(row)
        });

        container.innerHTML = '';
        container.appendChild(table.render());
    }

    renderPagination() {
        const container = document.getElementById('pagination-container');
        container.innerHTML = '';
        container.appendChild(this.pagination.render());
    }

    async showPromptDetails(prompt) {
        const modal = new Modal({ title: `Prompt #${prompt.id}`, size: 'large' });

        const successRate = Math.round(prompt.success_rate || 0);
        const rateColor = successRate >= 70 ? 'hsl(var(--success))' : successRate >= 40 ? 'hsl(142.1 70.6% 45.3%)' : 'hsl(var(--muted-foreground))';

        const content = document.createElement('div');

        // Fetch test results
        let testResultsHTML = '<div style="color: hsl(var(--muted-foreground)); text-align: center; padding: 2rem;">No test results available</div>';

        try {
            const response = await fetch(`/api/prompts/test-results/${prompt.id}`);
            const result = await response.json();

            if (result.success && result.data.length > 0) {
                testResultsHTML = result.data.map(test => {
                    const testSuccess = test.success ? 'Success' : 'Failed';
                    const testColor = test.success ? 'hsl(var(--success))' : 'hsl(var(--destructive))';
                    const severityColor = test.severity === 'critical' ? 'hsl(var(--destructive))' :
                                         test.severity === 'high' ? 'hsl(var(--warning))' :
                                         'hsl(var(--muted-foreground))';

                    const modelName = test.model || 'unknown';

                    return `
                        <div style="
                            padding: 1rem;
                            background: hsl(var(--card));
                            border: 1px solid hsl(var(--border));
                            border-radius: var(--radius-md);
                            margin-bottom: 1rem;
                        ">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <div style="display: flex; gap: 0.5rem; align-items: center;">
                                    <span style="font-weight: 600; color: ${testColor};">${testSuccess}</span>
                                    <span style="
                                        padding: 0.25rem 0.5rem;
                                        background: hsl(var(--muted) / 0.3);
                                        border-radius: var(--radius-sm);
                                        color: hsl(var(--muted-foreground));
                                        font-size: 0.8125rem;
                                        font-family: monospace;
                                    ">${test.provider} / ${modelName}</span>
                                </div>
                                <div style="display: flex; gap: 0.5rem; align-items: center;">
                                    ${test.severity ? `<span style="
                                        padding: 0.25rem 0.5rem;
                                        background: ${severityColor}33;
                                        color: ${severityColor};
                                        border-radius: var(--radius-sm);
                                        font-size: 0.75rem;
                                        font-weight: 600;
                                        text-transform: uppercase;
                                    ">${test.severity}</span>` : ''}
                                    ${test.confidence ? `<span style="
                                        padding: 0.25rem 0.5rem;
                                        background: hsl(var(--muted) / 0.3);
                                        color: hsl(var(--foreground));
                                        border-radius: var(--radius-sm);
                                        font-size: 0.75rem;
                                    ">${Math.round(test.confidence * 100)}% conf</span>` : ''}
                                </div>
                            </div>

                            ${test.response ? `
                                <div style="margin-bottom: 0.75rem;">
                                    <div style="font-size: 0.75rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem; font-weight: 600;">AI Response:</div>
                                    <div style="
                                        padding: 0.75rem;
                                        background: hsl(var(--muted) / 0.2);
                                        border-radius: var(--radius-sm);
                                        font-size: 0.875rem;
                                        line-height: 1.5;
                                        max-height: 150px;
                                        overflow-y: auto;
                                    ">${test.response.substring(0, 500)}${test.response.length > 500 ? '...' : ''}</div>
                                </div>
                            ` : ''}

                            ${test.reasoning ? `
                                <div>
                                    <div style="font-size: 0.75rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem; font-weight: 600;">Judge Reasoning:</div>
                                    <div style="
                                        padding: 0.75rem;
                                        background: hsl(var(--accent) / 0.3);
                                        border-left: 3px solid ${testColor};
                                        border-radius: var(--radius-sm);
                                        font-size: 0.875rem;
                                        line-height: 1.5;
                                    ">${test.reasoning}</div>
                                </div>
                            ` : ''}

                            <div style="margin-top: 0.75rem; font-size: 0.75rem; color: hsl(var(--muted-foreground));">
                                Tested: ${formatDate(test.tested_at)}
                                ${test.response_time ? ` • ${Math.round(test.response_time)}ms` : ''}
                            </div>
                        </div>
                    `;
                }).join('');
            }
        } catch (error) {
            console.error('Failed to load test results:', error);
        }

        content.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div style="
                    padding: 1.25rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-md);
                    text-align: center;
                ">
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem; font-weight: 600;">Success Rate</div>
                    <div style="font-size: 2rem; font-weight: 700; color: ${rateColor};">${successRate}%</div>
                </div>
                <div style="
                    padding: 1.25rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-md);
                    text-align: center;
                ">
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem; font-weight: 600;">Total Tests</div>
                    <div style="font-size: 2rem; font-weight: 700; color: hsl(var(--primary));">${prompt.test_count || 0}</div>
                </div>
                <div style="
                    padding: 1.25rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-md);
                    text-align: center;
                ">
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem; font-weight: 600;">Successes</div>
                    <div style="font-size: 2rem; font-weight: 700; color: hsl(var(--success));">${prompt.success_count || 0}</div>
                </div>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <strong style="font-size: 1rem;">Category:</strong>
                    ${createBadge(prompt.category, 'primary')}
                </div>
            </div>

            ${(prompt.purpose || prompt.risk_category || prompt.technique || prompt.modality) ? `
                <div style="margin-bottom: 1.5rem;">
                    <strong style="font-size: 1rem; display: block; margin-bottom: 0.75rem;">Classification:</strong>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        ${prompt.purpose ? `
                            <span style="
                                display: inline-flex;
                                align-items: center;
                                gap: 0.375rem;
                                padding: 0.375rem 0.75rem;
                                background: ${prompt.purpose === 'offensive' ? 'hsl(var(--destructive) / 0.15)' : 'hsl(var(--success) / 0.15)'};
                                color: ${prompt.purpose === 'offensive' ? 'hsl(var(--destructive))' : 'hsl(var(--success))'};
                                border: 1px solid ${prompt.purpose === 'offensive' ? 'hsl(var(--destructive) / 0.3)' : 'hsl(var(--success) / 0.3)'};
                                border-radius: var(--radius-md);
                                font-size: 0.875rem;
                                font-weight: 600;
                            ">
                                ${prompt.purpose === 'offensive' ? '⚔️ Offensive' : '🛡️ Defensive'}
                            </span>
                        ` : ''}
                        ${prompt.risk_category ? `
                            <span style="
                                display: inline-flex;
                                align-items: center;
                                gap: 0.375rem;
                                padding: 0.375rem 0.75rem;
                                background: hsl(var(--muted) / 0.3);
                                color: hsl(var(--foreground));
                                border: 1px solid hsl(var(--border));
                                border-radius: var(--radius-md);
                                font-size: 0.875rem;
                            ">
                                ${{
                                    'security': '🔒',
                                    'safety': '⚠️',
                                    'ethics': '🎭',
                                    'compliance': '📋',
                                    'misinformation': '📰'
                                }[prompt.risk_category] || '⚠️'} ${prompt.risk_category}
                            </span>
                        ` : ''}
                        ${prompt.technique ? `
                            <span style="
                                display: inline-flex;
                                align-items: center;
                                gap: 0.375rem;
                                padding: 0.375rem 0.75rem;
                                background: hsl(var(--accent) / 0.3);
                                color: hsl(var(--foreground));
                                border: 1px solid hsl(var(--border));
                                border-radius: var(--radius-md);
                                font-size: 0.875rem;
                            ">
                                🔧 ${prompt.technique.replace(/_/g, ' ')}
                            </span>
                        ` : ''}
                        ${prompt.modality ? `
                            <span style="
                                display: inline-flex;
                                align-items: center;
                                gap: 0.375rem;
                                padding: 0.375rem 0.75rem;
                                background: hsl(var(--muted) / 0.2);
                                color: hsl(var(--muted-foreground));
                                border: 1px solid hsl(var(--border));
                                border-radius: var(--radius-md);
                                font-size: 0.875rem;
                            ">
                                🎨 ${prompt.modality.replace(/_/g, ' ')}
                            </span>
                        ` : ''}
                    </div>
                </div>
            ` : ''}

            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <strong style="font-size: 1rem;">Payload:</strong>
                    <button id="copy-payload-btn" style="
                        display: inline-flex;
                        align-items: center;
                        gap: 0.375rem;
                        padding: 0.375rem 0.75rem;
                        background: hsl(var(--muted) / 0.3);
                        border: 1px solid hsl(var(--border));
                        border-radius: var(--radius-sm);
                        color: hsl(var(--foreground));
                        font-size: 0.8125rem;
                        font-weight: 500;
                        cursor: pointer;
                        transition: all 0.2s;
                    " onmouseover="this.style.background='hsl(var(--accent))'; this.style.borderColor='hsl(var(--primary) / 0.5)'" onmouseout="this.style.background='hsl(var(--muted) / 0.3)'; this.style.borderColor='hsl(var(--border))'">
                        <span>📋</span>
                        <span id="copy-btn-text">Copy</span>
                    </button>
                </div>
                <pre id="payload-text" style="
                    background: hsl(var(--muted) / 0.2);
                    padding: 1rem;
                    border-radius: var(--radius-md);
                    font-size: 0.9375rem;
                    font-family: 'Courier New', monospace;
                    line-height: 1.6;
                    border: 1px solid hsl(var(--border));
                    margin: 0;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                ">${prompt.payload}</pre>
            </div>

            ${prompt.description ? `
                <div style="margin-bottom: 1.5rem;">
                    <strong style="font-size: 1rem; display: block; margin-bottom: 0.5rem;">Description:</strong>
                    <p style="color: hsl(var(--muted-foreground)); line-height: 1.6;">${prompt.description}</p>
                </div>
            ` : ''}

            <div style="margin-bottom: 1.5rem;">
                <strong style="font-size: 1rem; display: block; margin-bottom: 0.5rem;">Source:</strong>
                <span style="color: hsl(var(--muted-foreground));">${prompt.source || 'N/A'}</span>
            </div>

            <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid hsl(var(--border));">
                <h4 style="font-size: 1.125rem; font-weight: 600; margin-bottom: 1rem;">Test Results (${prompt.test_count || 0})</h4>
                <div style="max-height: 500px; overflow-y: auto;">
                    ${testResultsHTML}
                </div>
            </div>
        `;

        modal.show(content);

        // Add copy functionality
        const copyBtn = content.querySelector('#copy-payload-btn');
        const copyBtnText = content.querySelector('#copy-btn-text');
        const payloadText = content.querySelector('#payload-text');

        if (copyBtn && payloadText) {
            copyBtn.addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(prompt.payload);
                    copyBtnText.textContent = 'Copied!';
                    copyBtn.style.background = 'hsl(var(--success) / 0.2)';
                    copyBtn.style.borderColor = 'hsl(var(--success))';

                    setTimeout(() => {
                        copyBtnText.textContent = 'Copy';
                        copyBtn.style.background = 'hsl(var(--muted) / 0.3)';
                        copyBtn.style.borderColor = 'hsl(var(--border))';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                    copyBtnText.textContent = 'Failed';
                    setTimeout(() => {
                        copyBtnText.textContent = 'Copy';
                    }, 2000);
                }
            });
        }
    }

    showAddPromptModal() {
        const modal = new Modal({ title: 'Add New Prompt' });

        const content = document.createElement('div');
        content.innerHTML = `
            <form id="add-prompt-form">
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Category</label>
                    <select name="category" required style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                        <option value="jailbreak">Jailbreak</option>
                        <option value="injection">Prompt Injection</option>
                        <option value="harmful">Harmful Behavior</option>
                        <option value="other">Other</option>
                    </select>
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Payload</label>
                    <textarea name="payload" required rows="5" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                        font-family: monospace;
                        resize: vertical;
                    "></textarea>
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Description (optional)</label>
                    <input type="text" name="description" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                </div>
                <div class="flex gap-1 mt-2">
                    <button type="submit" class="btn btn-primary">Add Prompt</button>
                    <button type="button" class="btn btn-secondary cancel-btn">Cancel</button>
                </div>
            </form>
        `;

        content.querySelector('.cancel-btn').addEventListener('click', () => modal.close());

        content.querySelector('#add-prompt-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);

            try {
                const response = await fetch('/api/prompts/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        category: formData.get('category'),
                        payload: formData.get('payload'),
                        description: formData.get('description'),
                        source: 'manual'
                    })
                });

                const result = await response.json();

                if (result.success) {
                    modal.close();
                    this.loadPrompts(this.currentPage);
                    Modal.alert('Success', 'Prompt added successfully!');
                }
            } catch (error) {
                console.error('Failed to add prompt:', error);
                Modal.alert('Error', 'Failed to add prompt');
            }
        });

        modal.show(content);
    }

    async deletePrompt(promptId) {
        try {
            const response = await fetch(`/api/prompts/${promptId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (result.success) {
                this.loadPrompts(this.currentPage);
                Modal.alert('Success', 'Prompt deleted successfully!');
            }
        } catch (error) {
            console.error('Failed to delete prompt:', error);
            Modal.alert('Error', 'Failed to delete prompt');
        }
    }

    getCategoryIcon(category) {
        const icons = {
            'jailbreak': '🔓',
            'injection': '💉',
            'harmful': '⚠️',
            'bias': '⚖️',
            'privacy': '🔒',
            'other': '📝'
        };
        return icons[category] || '📝';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new PromptsPage();
});
