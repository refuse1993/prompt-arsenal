/**
 * Multimodal Page Module
 * Multimodal attack management
 */

import { Card } from '../components/Card.js';
import { Table, createBadge, formatDate } from '../components/Table.js';
import { Pagination } from '../components/Pagination.js';
import { Modal } from '../components/Modal.js';
import { Chart } from '../components/Chart.js';

class MultimodalPage {
    constructor() {
        this.currentPage = 1;
        this.currentMediaType = null;
        this.currentAttackType = null;
        this.currentModel = null;
        this.pagination = new Pagination({
            currentPage: 1,
            totalPages: 1,
            onPageChange: (page) => this.loadMedia(page)
        });

        this.init();
    }

    async init() {
        await this.loadMediaTypes();
        await this.loadAttackTypes();
        await this.renderFilters();
        await this.loadMedia();
    }

    async loadMediaTypes() {
        try {
            const response = await fetch('/api/multimodal/types');
            const result = await response.json();

            if (result.success) {
                this.renderMediaTypes(result.data);
            }
        } catch (error) {
            console.error('Failed to load media types:', error);
        }
    }

    renderMediaTypes(types) {
        const container = document.getElementById('media-types-container');

        const mediaTypes = [
            { name: 'image', icon: '🖼️', color: '#3b82f6' },
            { name: 'audio', icon: '🎵', color: '#8b5cf6' },
            { name: 'video', icon: '🎬', color: '#10b981' }
        ];

        mediaTypes.forEach(type => {
            const count = types.find(t => t.media_type === type.name)?.count || 0;
            const card = Card.createStatCard(count, type.name.toUpperCase(), type.icon);

            card.style.cursor = 'pointer';
            card.addEventListener('click', () => {
                this.currentMediaType = type.name;
                this.loadMedia(1);
            });

            container.appendChild(card);
        });
    }

    async loadAttackTypes() {
        try {
            const response = await fetch('/api/multimodal/attack-types');
            const result = await response.json();

            if (result.success) {
                this.attackTypes = result.data;
                this.renderAttackTypesChart(result.data);
                this.renderSuccessRatesChart();
            }
        } catch (error) {
            console.error('Failed to load attack types:', error);
        }
    }

    renderAttackTypesChart(attackTypes) {
        const container = document.getElementById('attack-types-chart');

        const chartData = attackTypes.map(type => ({
            label: type.attack_type,
            value: type.count,
            color: this.getAttackTypeColor(type.attack_type)
        }));

        container.appendChild(Chart.createPieChart(chartData));
    }

    async renderSuccessRatesChart() {
        const container = document.getElementById('success-rates-chart');

        try {
            const response = await fetch('/api/multimodal/stats/success-rates');
            const result = await response.json();

            if (result.success && result.data.length > 0) {
                const chartData = result.data.map(item => ({
                    label: item.type,
                    value: item.rate,
                    color: this.getAttackTypeColor(item.type)
                }));

                container.appendChild(Chart.createBarChart(chartData));
            } else {
                // 데이터가 없으면 "No data" 메시지
                container.innerHTML = '<div style="text-align: center; padding: 2rem; color: hsl(var(--muted-foreground));">데이터 없음</div>';
            }
        } catch (error) {
            console.error('Failed to load success rates:', error);
            // 에러 발생 시에도 더미 데이터 표시하지 않음
            container.innerHTML = '<div style="text-align: center; padding: 2rem; color: hsl(var(--muted-foreground));">데이터를 불러올 수 없습니다</div>';
        }
    }

    // 이전 더미 데이터 버전 (백업용, 사용 안함)
    renderSuccessRatesChartDummy() {
        const container = document.getElementById('success-rates-chart');

        const chartData = [
            { label: 'FGSM', value: 75, color: '#3b82f6' },
            { label: 'PGD', value: 82, color: '#8b5cf6' },
            { label: 'C&W', value: 68, color: '#10b981' },
            { label: 'Pixel', value: 55, color: '#f59e0b' }
        ];

        container.appendChild(Chart.createBarChart(chartData));
    }

    async renderFilters() {
        const container = document.getElementById('filter-container');

        // Get available models
        let modelOptions = '<option value="">All Models</option>';
        try {
            const response = await fetch('/api/multimodal/models');
            const result = await response.json();

            if (result.success && result.data.length > 0) {
                modelOptions = '<option value="">All Models</option>';
                result.data.forEach(item => {
                    modelOptions += `<option value="${item.provider}/${item.model}">${item.provider} / ${item.model} (${item.count})</option>`;
                });
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }

        container.innerHTML = `
            <select id="media-type-filter" style="
                padding: var(--spacing-sm) var(--spacing-md);
                background: var(--bg-hover);
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                color: var(--text-primary);
            ">
                <option value="">All Media Types</option>
                <option value="image">Image</option>
                <option value="audio">Audio</option>
                <option value="video">Video</option>
            </select>

            <select id="attack-type-filter" style="
                padding: var(--spacing-sm) var(--spacing-md);
                background: var(--bg-hover);
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                color: var(--text-primary);
            ">
                <option value="">All Attack Types</option>
                <option value="fgsm">FGSM</option>
                <option value="pgd">PGD</option>
                <option value="cw">C&W</option>
                <option value="pixel">Pixel Attack</option>
                <option value="ultrasonic">Ultrasonic</option>
            </select>

            <select id="model-filter" style="
                padding: var(--spacing-sm) var(--spacing-md);
                background: var(--bg-hover);
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                color: var(--text-primary);
            ">
                ${modelOptions}
            </select>
        `;

        document.getElementById('media-type-filter').addEventListener('change', (e) => {
            this.currentMediaType = e.target.value || null;
            this.loadMedia(1);
        });

        document.getElementById('attack-type-filter').addEventListener('change', (e) => {
            this.currentAttackType = e.target.value || null;
            this.loadMedia(1);
        });

        document.getElementById('model-filter').addEventListener('change', (e) => {
            this.currentModel = e.target.value || null;
            this.loadMedia(1);
        });
    }

    async loadMedia(page = 1) {
        this.currentPage = page;

        try {
            let url = `/api/multimodal/list?page=${page}&limit=50`;
            if (this.currentMediaType) {
                url += `&media_type=${this.currentMediaType}`;
            }
            if (this.currentAttackType) {
                url += `&attack_type=${this.currentAttackType}`;
            }
            if (this.currentModel) {
                url += `&model=${encodeURIComponent(this.currentModel)}`;
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
            console.error('Failed to load media:', error);
        }
    }

    renderTable(media) {
        const container = document.getElementById('table-container');

        const table = new Table({
            columns: [
                {
                    key: 'id',
                    label: 'ID',
                    sortable: true
                },
                {
                    key: 'media_type',
                    label: 'Type',
                    sortable: true,
                    render: (value) => {
                        const icons = { image: '🖼️', audio: '🎵', video: '🎬' };
                        return `${icons[value] || ''} ${createBadge(value, 'primary')}`;
                    }
                },
                {
                    key: 'attack_type',
                    label: 'Attack',
                    sortable: true,
                    render: (value) => createBadge(value, 'secondary')
                },
                {
                    key: 'base_file',
                    label: 'Base File',
                    sortable: false,
                    render: (value) => {
                        const filename = value.split('/').pop();
                        return `<span style="font-family: monospace; font-size: 0.875rem;">${filename}</span>`;
                    }
                },
                {
                    key: 'success_rate',
                    label: 'Success Rate',
                    sortable: true,
                    render: (value) => {
                        if (!value) return '-';
                        const color = value > 70 ? 'success' : value > 50 ? 'warning' : 'danger';
                        return createBadge(`${value}%`, color);
                    }
                },
                {
                    key: 'created_at',
                    label: 'Created',
                    sortable: true,
                    render: formatDate
                }
            ],
            data: media,
            onRowClick: (row) => this.showMediaDetails(row)
        });

        container.innerHTML = '';
        container.appendChild(table.render());
    }

    renderPagination() {
        const container = document.getElementById('pagination-container');
        container.innerHTML = '';
        container.appendChild(this.pagination.render());
    }

    async showMediaDetails(media) {
        const modal = new Modal({ title: `${media.media_type.toUpperCase()} Attack #${media.id}`, size: 'large' });

        const content = document.createElement('div');

        // Fetch test results
        let testResultsHTML = '';
        let testCount = 0;
        let successCount = 0;
        let successRate = 0;

        try {
            const response = await fetch(`/api/multimodal/test-results/${media.id}`);
            const result = await response.json();

            if (result.success && result.data.length > 0) {
                testCount = result.data.length;
                successCount = result.data.filter(t => t.success).length;
                successRate = Math.round((successCount / testCount) * 100);

                testResultsHTML = result.data.map(test => {
                    const testSuccess = test.success ? 'Success' : 'Failed';
                    const testColor = test.success ? 'hsl(var(--success))' : 'hsl(var(--destructive))';
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
                                <div style="font-size: 0.75rem; color: hsl(var(--muted-foreground));">
                                    ${formatDate(test.tested_at)}
                                </div>
                            </div>

                            ${test.vision_response ? `
                                <div>
                                    <div style="font-size: 0.75rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem; font-weight: 600;">Vision Response:</div>
                                    <div style="
                                        padding: 0.75rem;
                                        background: hsl(var(--muted) / 0.2);
                                        border-radius: var(--radius-sm);
                                        font-size: 0.875rem;
                                        line-height: 1.5;
                                        max-height: 150px;
                                        overflow-y: auto;
                                    ">${test.vision_response.substring(0, 500)}${test.vision_response.length > 500 ? '...' : ''}</div>
                                </div>
                            ` : ''}
                        </div>
                    `;
                }).join('');
            } else {
                testResultsHTML = '<div style="color: hsl(var(--muted-foreground)); text-align: center; padding: 2rem;">No test results available</div>';
            }
        } catch (error) {
            console.error('Failed to load test results:', error);
            testResultsHTML = '<div style="color: hsl(var(--muted-foreground)); text-align: center; padding: 2rem;">Failed to load test results</div>';
        }

        // Parse parameters
        const params = JSON.parse(media.parameters || '{}');
        const paramEntries = Object.entries(params);

        // 미디어 타입별 렌더링
        let mediaPreview = '';
        const generatedPath = media.generated_file;

        if (media.media_type === 'image') {
            mediaPreview = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                    <div>
                        <div style="font-weight: 600; margin-bottom: 0.5rem; color: hsl(var(--foreground));">Base Image</div>
                        <img src="/${media.base_file}" alt="Base" style="width: 100%; border-radius: var(--radius-md); border: 1px solid hsl(var(--border));" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div style="display: none; padding: 2rem; text-align: center; background: hsl(var(--muted) / 0.2); border-radius: var(--radius-md); color: hsl(var(--muted-foreground));">
                            Image not found
                        </div>
                    </div>
                    <div>
                        <div style="font-weight: 600; margin-bottom: 0.5rem; color: hsl(var(--foreground));">Generated Attack</div>
                        <img src="/${generatedPath}" alt="Generated" style="width: 100%; border-radius: var(--radius-md); border: 1px solid hsl(var(--border));" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div style="display: none; padding: 2rem; text-align: center; background: hsl(var(--muted) / 0.2); border-radius: var(--radius-md); color: hsl(var(--muted-foreground));">
                            Image not found
                        </div>
                    </div>
                </div>
            `;
        } else if (media.media_type === 'audio') {
            mediaPreview = `
                <div style="margin-bottom: 1.5rem;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: hsl(var(--foreground));">Base Audio</div>
                    <audio controls style="width: 100%; margin-bottom: 1rem;">
                        <source src="/${media.base_file}" type="audio/wav">
                        Your browser does not support audio.
                    </audio>
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: hsl(var(--foreground));">Generated Attack</div>
                    <audio controls style="width: 100%;">
                        <source src="/${generatedPath}" type="audio/wav">
                        Your browser does not support audio.
                    </audio>
                </div>
            `;
        } else if (media.media_type === 'video') {
            mediaPreview = `
                <div style="margin-bottom: 1.5rem;">
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: hsl(var(--foreground));">Base Video</div>
                    <video controls style="width: 100%; max-height: 300px; margin-bottom: 1rem; border-radius: var(--radius-md);">
                        <source src="/${media.base_file}" type="video/mp4">
                        Your browser does not support video.
                    </video>
                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: hsl(var(--foreground));">Generated Attack</div>
                    <video controls style="width: 100%; max-height: 300px; border-radius: var(--radius-md);">
                        <source src="/${generatedPath}" type="video/mp4">
                        Your browser does not support video.
                    </video>
                </div>
            `;
        }

        const rateColor = successRate >= 70 ? 'hsl(var(--success))' : successRate >= 40 ? 'hsl(142.1 70.6% 45.3%)' : 'hsl(var(--muted-foreground))';

        content.innerHTML = `
            ${mediaPreview}

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
                    <div style="font-size: 2rem; font-weight: 700; color: hsl(var(--primary));">${testCount}</div>
                </div>
                <div style="
                    padding: 1.25rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-md);
                    text-align: center;
                ">
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem; font-weight: 600;">Successes</div>
                    <div style="font-size: 2rem; font-weight: 700; color: hsl(var(--success));">${successCount}</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Media Type</div>
                    <div>${createBadge(media.media_type, 'primary')}</div>
                </div>
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Attack Type</div>
                    <div>${createBadge(media.attack_type, 'secondary')}</div>
                </div>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Base File</div>
                <code style="
                    display: block;
                    padding: 0.75rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-sm);
                    font-size: 0.8125rem;
                    word-break: break-all;
                ">${media.base_file}</code>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Generated File</div>
                <code style="
                    display: block;
                    padding: 0.75rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-sm);
                    font-size: 0.8125rem;
                    word-break: break-all;
                ">${generatedPath}</code>
            </div>

            ${paramEntries.length > 0 ? `
                <div style="margin-bottom: 1.5rem;">
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.75rem;">Attack Parameters</div>
                    <div style="display: grid; gap: 0.75rem;">
                        ${paramEntries.map(([key, value]) => `
                            <div style="
                                padding: 0.75rem;
                                background: hsl(var(--card));
                                border: 1px solid hsl(var(--border));
                                border-radius: var(--radius-sm);
                            ">
                                <div style="font-weight: 600; font-size: 0.75rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem; text-transform: uppercase;">${key}</div>
                                <div style="font-family: monospace; font-size: 0.875rem; color: hsl(var(--foreground)); word-break: break-all;">
                                    ${typeof value === 'string' && value.length > 100 ? value.substring(0, 100) + '...' : JSON.stringify(value)}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid hsl(var(--border));">
                <h4 style="font-size: 1.125rem; font-weight: 600; margin-bottom: 1rem;">Test Results (${testCount})</h4>
                <div style="max-height: 500px; overflow-y: auto;">
                    ${testResultsHTML}
                </div>
            </div>
        `;

        modal.show(content);
    }

    showGenerateAttackModal() {
        const modal = new Modal({ title: 'Generate Adversarial Attack' });

        const content = document.createElement('div');
        content.innerHTML = `
            <form id="generate-attack-form">
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Media Type</label>
                    <select name="media_type" required style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                        <option value="image">🖼️ Image</option>
                        <option value="audio">🎵 Audio</option>
                        <option value="video">🎬 Video</option>
                    </select>
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Attack Type</label>
                    <select name="attack_type" required style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                        <option value="fgsm">FGSM - Fast Gradient Sign Method</option>
                        <option value="pgd">PGD - Projected Gradient Descent</option>
                        <option value="cw">C&W - Carlini & Wagner</option>
                        <option value="pixel">Pixel Attack</option>
                    </select>
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Base File Path</label>
                    <input type="text" name="base_file" required placeholder="media/original.png" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                </div>
                <div class="flex gap-1 mt-2">
                    <button type="submit" class="btn btn-primary">⚡ Generate</button>
                    <button type="button" class="btn btn-secondary cancel-btn">Cancel</button>
                </div>
            </form>
        `;

        content.querySelector('.cancel-btn').addEventListener('click', () => modal.close());

        content.querySelector('#generate-attack-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            modal.close();
            Modal.alert('Success', 'Attack generation started! Check the media arsenal for results.');
        });

        modal.show(content);
    }

    async deleteMedia(mediaId) {
        try {
            const response = await fetch(`/api/multimodal/${mediaId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (result.success) {
                this.loadMedia(this.currentPage);
                Modal.alert('Success', 'Media deleted successfully!');
            }
        } catch (error) {
            console.error('Failed to delete media:', error);
            Modal.alert('Error', 'Failed to delete media');
        }
    }

    getAttackTypeColor(type) {
        const colors = {
            'fgsm': '#3b82f6',
            'pgd': '#8b5cf6',
            'cw': '#10b981',
            'pixel': '#f59e0b',
            'ultrasonic': '#06b6d4'
        };
        return colors[type] || '#64748b';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new MultimodalPage();
});
