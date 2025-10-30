/**
 * Multi-turn Page Module
 * Multi-turn campaign management
 */

import { Card } from '../components/Card.js';
import { Table, createBadge, formatDate } from '../components/Table.js';
import { Modal } from '../components/Modal.js';
import { Chart } from '../components/Chart.js';

class MultiturnPage {
    constructor() {
        this.init();
    }

    async init() {
        await this.loadStrategies();
        await this.loadCampaigns();
        await this.loadTestResults();
    }

    async loadStrategies() {
        try {
            const response = await fetch('/api/multiturn/strategies');
            const result = await response.json();

            if (result.success) {
                this.strategies = result.data;
                this.renderStrategies(result.data);
                this.renderSuccessChart(result.data);
            }
        } catch (error) {
            console.error('Failed to load strategies:', error);
        }
    }

    renderStrategies(strategies) {
        const container = document.getElementById('strategies-container');
        container.innerHTML = '';

        if (strategies.length === 0) {
            container.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: hsl(var(--muted-foreground));">전략 데이터 없음</div>';
            return;
        }

        strategies.forEach(strategy => {
            const actualRate = strategy.actual_success_rate * 100;
            const paperRate = strategy.paper_asr * 100;
            const isWorking = actualRate > 0;

            const card = document.createElement('div');
            card.className = 'card';
            card.style.cursor = 'pointer';
            card.style.transition = 'transform 0.2s, box-shadow 0.2s';
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-2px)';
                card.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            });
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = '';
            });

            card.innerHTML = `
                <div class="card-body">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                        <div style="font-size: 2rem;">${this.getStrategyIcon(strategy.name)}</div>
                        <div style="flex: 1;">
                            <h3 style="font-size: 1.125rem; font-weight: 600; margin: 0;">${strategy.name}</h3>
                            <p style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin: 0.25rem 0 0 0;">
                                ${strategy.description}
                            </p>
                        </div>
                    </div>

                    <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem;">
                        <div style="flex: 1; text-align: center; padding: 0.5rem; background: hsl(var(--muted) / 0.2); border-radius: var(--radius-sm);">
                            <div style="font-size: 0.75rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem;">Paper ASR</div>
                            <div style="font-size: 1.25rem; font-weight: 600; color: hsl(var(--primary));">${paperRate.toFixed(1)}%</div>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 0.5rem; background: ${isWorking ? 'hsl(var(--success) / 0.1)' : 'hsl(var(--muted) / 0.2)'}; border-radius: var(--radius-sm);">
                            <div style="font-size: 0.75rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.25rem;">Actual</div>
                            <div style="font-size: 1.25rem; font-weight: 600; color: ${isWorking ? 'hsl(var(--success))' : 'hsl(var(--muted-foreground))'};">${actualRate.toFixed(1)}%</div>
                        </div>
                    </div>

                    <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 0.75rem; border-top: 1px solid hsl(var(--border));">
                        <span style="font-size: 0.875rem; color: hsl(var(--muted-foreground));">
                            ${strategy.campaigns} campaigns • ${strategy.evaluations} evals
                        </span>
                        ${isWorking ? '<span style="color: hsl(var(--success)); font-size: 0.875rem;">✓ Active</span>' : '<span style="color: hsl(var(--muted-foreground)); font-size: 0.875rem;">○ Inactive</span>'}
                    </div>
                </div>
            `;

            container.appendChild(card);
        });
    }

    renderSuccessChart(strategies) {
        const container = document.getElementById('success-chart');
        container.innerHTML = '';

        if (strategies.length === 0) {
            container.innerHTML = '<div style="text-align: center; padding: 2rem; color: hsl(var(--muted-foreground));">차트 데이터 없음</div>';
            return;
        }

        // 논문 ASR과 실제 성공률 비교 차트
        const chartData = strategies.map(s => ({
            label: s.name,
            value: s.paper_asr * 100,  // 논문 데이터
            value2: s.actual_success_rate * 100,  // 실제 데이터
            color: this.getStrategyColor(s.name)
        }));

        container.appendChild(Chart.createBarChart(chartData));
    }

    async loadCampaigns() {
        try {
            const response = await fetch('/api/multiturn/campaigns');
            const result = await response.json();

            if (result.success) {
                this.renderCampaigns(result.data);
            }
        } catch (error) {
            console.error('Failed to load campaigns:', error);
        }
    }

    renderCampaigns(campaigns) {
        const container = document.getElementById('campaigns-container');

        if (campaigns.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No campaigns yet. Create your first campaign!</p>';
            return;
        }

        const table = new Table({
            columns: [
                {
                    key: 'id',
                    label: 'ID',
                    sortable: true
                },
                {
                    key: 'name',
                    label: 'Campaign Name',
                    sortable: true
                },
                {
                    key: 'strategy',
                    label: 'Strategy',
                    sortable: true,
                    render: (value) => {
                        const icon = this.getStrategyIcon(value);
                        return `${icon} ${createBadge(value, 'primary')}`;
                    }
                },
                {
                    key: 'steps',
                    label: 'Steps',
                    sortable: true
                },
                {
                    key: 'success_rate',
                    label: 'Success Rate',
                    sortable: true,
                    render: (value, row) => {
                        if (row.total_evaluations === 0) {
                            return '<span style="color: hsl(var(--muted-foreground)); font-size: 0.875rem;">No evals</span>';
                        }
                        const rate = Math.round(value || 0);
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
                    key: 'created_at',
                    label: 'Created',
                    sortable: true,
                    render: formatDate
                }
            ],
            data: campaigns,
            onRowClick: (row) => this.showCampaignDetails(row)
        });

        container.innerHTML = '';
        container.appendChild(table.render());
    }

    async loadTestResults() {
        try {
            const response = await fetch('/api/multiturn/test-results');
            const result = await response.json();

            if (result.success) {
                this.renderTestResults(result.data);
            }
        } catch (error) {
            console.error('Failed to load test results:', error);
        }
    }

    renderTestResults(results) {
        const container = document.getElementById('results-container');
        container.innerHTML = '';

        if (results.length === 0) {
            container.innerHTML = '<p style="color: hsl(var(--muted-foreground));">평가 결과 없음</p>';
            return;
        }

        const table = new Table({
            columns: [
                {
                    key: 'campaign_id',
                    label: 'Campaign ID',
                    sortable: true
                },
                {
                    key: 'campaign_name',
                    label: 'Name',
                    sortable: true,
                    render: (value, row) => {
                        const icon = this.getStrategyIcon(row.strategy);
                        return `${icon} ${value || 'Unnamed'}`;
                    }
                },
                {
                    key: 'strategy',
                    label: 'Strategy',
                    sortable: true,
                    render: (value) => createBadge(value, 'primary')
                },
                {
                    key: 'model',
                    label: 'Model',
                    sortable: true
                },
                {
                    key: 'turns_used',
                    label: 'Turns',
                    sortable: true
                },
                {
                    key: 'total_evaluations',
                    label: 'Evaluations',
                    sortable: true
                },
                {
                    key: 'avg_progress',
                    label: 'Avg Progress',
                    sortable: true,
                    render: (value) => {
                        const color = value > 70 ? 'success' : value > 50 ? 'warning' : 'danger';
                        return createBadge(`${value}%`, color);
                    }
                },
                {
                    key: 'success_rate',
                    label: 'Success Rate',
                    sortable: true,
                    render: (value, row) => {
                        if (row.total_evaluations === 0) {
                            return '<span style="color: hsl(var(--muted-foreground)); font-size: 0.875rem;">No evals</span>';
                        }
                        const rate = Math.round(value || 0);
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
                    key: 'status',
                    label: 'Status',
                    sortable: true,
                    render: (value) => {
                        const colorMap = {
                            'completed': 'success',
                            'running': 'warning',
                            'failed': 'danger',
                            'pending': 'secondary'
                        };
                        return createBadge(value, colorMap[value] || 'secondary');
                    }
                },
                {
                    key: 'created_at',
                    label: 'Created',
                    sortable: true,
                    render: formatDate
                }
            ],
            data: results
        });

        container.appendChild(table.render());
    }

    async showCampaignDetails(campaign) {
        const modal = new Modal({ title: `Campaign: ${campaign.name || 'Unnamed'}` });

        try {
            const response = await fetch(`/api/multiturn/campaigns/${campaign.id}`);
            const result = await response.json();

            if (result.success) {
                const { campaign: details, conversations, evaluations } = result.data;

                const content = document.createElement('div');
                content.innerHTML = `
                    <div class="mb-1">
                        <strong>Strategy:</strong> ${this.getStrategyIcon(details.strategy)} ${createBadge(details.strategy, 'primary')}
                    </div>
                    <div class="mb-1">
                        <strong>Model:</strong> ${details.target_model || 'N/A'}
                    </div>
                    <div class="mb-1">
                        <strong>Status:</strong> ${createBadge(details.status, details.status === 'completed' ? 'success' : 'warning')}
                    </div>
                    <div class="mb-1">
                        <strong>Total Turns:</strong> ${conversations.length}
                    </div>
                    <div class="mb-1">
                        <strong>Goal:</strong> ${details.goal || 'N/A'}
                    </div>
                    <div class="mb-1">
                        <strong>Conversation History:</strong>
                        <div style="margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.5rem; max-height: 400px; overflow-y: auto;">
                            ${conversations.map((conv, idx) => {
                                const eval_data = evaluations.find(e => e.turn_number === conv.turn_number);
                                return `
                                <div style="
                                    background: hsl(var(--card));
                                    border: 1px solid hsl(var(--border));
                                    padding: var(--spacing-md);
                                    border-radius: var(--radius-md);
                                ">
                                    <div style="font-weight: 600; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                                        <span>Turn ${conv.turn_number + 1}</span>
                                        ${eval_data ? this.renderProgressIndicator(eval_data) : ''}
                                    </div>
                                    <div style="font-size: 0.875rem; margin-bottom: 0.5rem;">
                                        <strong>Prompt:</strong>
<div style="color: hsl(var(--muted-foreground)); margin-top: 0.25rem; padding: 0.5rem; background: hsl(var(--muted) / 0.1); border-radius: 4px; max-height: 120px; overflow-y: auto; line-height: 1.5; white-space: pre-wrap; word-break: break-word;">${this.cleanText(conv.prompt_text)}</div>
                                    </div>
                                    <div style="font-size: 0.875rem;">
                                        <strong>Response:</strong>
<div style="color: hsl(var(--muted-foreground)); margin-top: 0.25rem; padding: 0.5rem; background: hsl(var(--muted) / 0.1); border-radius: 4px; max-height: 120px; overflow-y: auto; line-height: 1.5; white-space: pre-wrap; word-break: break-word;">${this.cleanText(conv.response)}</div>
                                    </div>
                                </div>
                            `}).join('')}
                        </div>
                    </div>
                `;

                modal.show(content);
            }
        } catch (error) {
            console.error('Failed to load campaign details:', error);
        }
    }

    showResultDetails(result) {
        const modal = new Modal({ title: `Test Result #${result.id}` });

        const content = document.createElement('div');
        content.innerHTML = `
            <div class="mb-1">
                <strong>Campaign ID:</strong> ${result.campaign_id}
            </div>
            <div class="mb-1">
                <strong>Model:</strong> ${result.model_name}
            </div>
            <div class="mb-1">
                <strong>Result:</strong> ${result.success
                    ? createBadge('Success', 'success')
                    : createBadge('Failed', 'danger')}
            </div>
            <div class="mb-1">
                <strong>Turns Completed:</strong> ${result.turns_completed}
            </div>
            <div class="mb-1">
                <strong>Tested:</strong> ${formatDate(result.tested_at)}
            </div>
        `;

        modal.show(content);
    }

    showCreateCampaignModal() {
        const modal = new Modal({ title: 'Create Multi-turn Campaign' });

        const content = document.createElement('div');
        content.innerHTML = `
            <form id="create-campaign-form">
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Campaign Name</label>
                    <input type="text" name="name" required placeholder="My Campaign" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Strategy</label>
                    <select name="strategy" required style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                        ${this.strategies.map(s => `
                            <option value="${s.name}">${s.name} (${(s.avg_success_rate * 100).toFixed(1)}% ASR)</option>
                        `).join('')}
                    </select>
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Description</label>
                    <textarea name="description" rows="3" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                        resize: vertical;
                    "></textarea>
                </div>
                <div class="flex gap-1 mt-2">
                    <button type="submit" class="btn btn-primary">Create Campaign</button>
                    <button type="button" class="btn btn-secondary cancel-btn">Cancel</button>
                </div>
            </form>
        `;

        content.querySelector('.cancel-btn').addEventListener('click', () => modal.close());

        content.querySelector('#create-campaign-form').addEventListener('submit', (e) => {
            e.preventDefault();
            modal.close();
            Modal.alert('Success', 'Campaign created successfully!');
            this.loadCampaigns();
        });

        modal.show(content);
    }

    getStrategyIcon(strategy) {
        const icons = {
            'FigStep': '📊',
            'Crescendo': '📈',
            'RolePlay': '🎭',
            'Crescendobis': '📊',
            'Latent': '🧠',
            'Seed': '🌱',
            'Deception': '🎪'
        };
        return icons[strategy] || '🔄';
    }

    getStrategyColor(strategy) {
        const colors = {
            'FigStep': '#3b82f6',
            'Crescendo': '#8b5cf6',
            'RolePlay': '#10b981',
            'Crescendobis': '#f59e0b',
            'Latent': '#ef4444',
            'Seed': '#06b6d4',
            'Deception': '#ec4899'
        };
        return colors[strategy] || '#64748b';
    }

    cleanText(text) {
        if (!text) return 'N/A';

        // 1. 각 줄의 앞뒤 공백 제거
        // 2. 빈 줄 제거
        // 3. 여러 개의 연속된 공백을 하나로
        return text
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .join('\n')
            .replace(/  +/g, ' ');
    }

    renderProgressIndicator(evalData) {
        const progress = (evalData.progress * 100).toFixed(0);
        const goalAchieved = evalData.goal_achieved;

        // 진행상황에 따른 색상 및 아이콘 결정
        let color, bgColor, icon, statusText;

        if (goalAchieved) {
            // 목표 달성 - 녹색
            color = 'hsl(var(--success))';
            bgColor = 'hsl(142.1 76.2% 36.3% / 0.1)';
            icon = '✅';
            statusText = 'Goal Achieved';
        } else if (progress >= 70) {
            // 높은 진행률 - 주황색 (거의 성공)
            color = 'hsl(24.6 95% 53.1%)';
            bgColor = 'hsl(24.6 95% 53.1% / 0.1)';
            icon = '⚠️';
            statusText = `${progress}% Progress`;
        } else if (progress >= 30) {
            // 중간 진행률 - 파란색
            color = 'hsl(221.2 83.2% 53.3%)';
            bgColor = 'hsl(221.2 83.2% 53.3% / 0.1)';
            icon = '🔵';
            statusText = `${progress}% Progress`;
        } else if (progress > 0) {
            // 낮은 진행률 - 회색
            color = 'hsl(var(--muted-foreground))';
            bgColor = 'hsl(var(--muted) / 0.3)';
            icon = '⚪';
            statusText = `${progress}% Progress`;
        } else {
            // 실패 - 빨간색
            color = 'hsl(var(--destructive))';
            bgColor = 'hsl(var(--destructive) / 0.1)';
            icon = '❌';
            statusText = 'Failed';
        }

        return `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 0.375rem;
                    padding: 0.25rem 0.5rem;
                    background: ${bgColor};
                    border-radius: var(--radius-md);
                    font-size: 0.75rem;
                    font-weight: 600;
                    color: ${color};
                    border: 1px solid ${color};
                ">
                    <span>${icon}</span>
                    <span>${statusText}</span>
                </div>
                ${goalAchieved ? '' : `
                    <div style="
                        width: 60px;
                        height: 8px;
                        background: hsl(var(--muted) / 0.3);
                        border-radius: 4px;
                        overflow: hidden;
                    ">
                        <div style="
                            height: 100%;
                            width: ${progress}%;
                            background: ${color};
                            transition: width 0.3s;
                        "></div>
                    </div>
                `}
            </div>
        `;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new MultiturnPage();
});
