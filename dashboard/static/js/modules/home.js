/**
 * Home Page Module
 * Dashboard overview and statistics
 */

import { Card } from '../components/Card.js';
import { Chart as ChartComponent } from '../components/Chart.js';

class HomePage {
    constructor() {
        this.loadStats();
        this.loadClassificationStats();
        this.loadActivity();
        this.loadSuccessRates();
    }

    async loadStats() {
        try {
            const response = await fetch('/api/stats/overview');
            const result = await response.json();

            if (result.success) {
                this.renderStats(result.data);
            }
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    renderStats(data) {
        const container = document.getElementById('stats-container');
        const grid = document.createElement('div');
        grid.className = 'grid grid-4';

        // Text Prompts
        grid.appendChild(Card.createStatCard(
            data.prompts.total,
            'Text Prompts',
            '📝'
        ));

        // Multimodal
        grid.appendChild(Card.createStatCard(
            data.multimodal.total,
            'Multimodal Attacks',
            '🎨'
        ));

        // CTF
        grid.appendChild(Card.createStatCard(
            data.ctf.challenges,
            'CTF Challenges',
            '🎯'
        ));

        // Total Tests
        const totalTests = data.prompts.tests + data.multimodal.tests + data.ctf.tests;
        grid.appendChild(Card.createStatCard(
            totalTests,
            'Total Tests',
            '⚡'
        ));

        container.appendChild(grid);
    }

    async loadActivity() {
        try {
            const response = await fetch('/api/stats/activity');
            const result = await response.json();

            if (result.success) {
                this.renderActivity(result.data);
            }
        } catch (error) {
            console.error('Failed to load activity:', error);
        }
    }

    renderActivity(activities) {
        const container = document.getElementById('activity-container');

        if (activities.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No recent activity</p>';
            return;
        }

        const list = document.createElement('div');
        list.style.cssText = 'display: flex; flex-direction: column; gap: var(--spacing-md);';

        activities.slice(0, 10).forEach(activity => {
            const item = document.createElement('div');
            item.style.cssText = `
                display: flex;
                justify-content: space-between;
                padding: var(--spacing-sm);
                border-radius: var(--radius-md);
                background: var(--bg-hover);
            `;

            const icon = this.getActivityIcon(activity.type);
            const badge = activity.success
                ? '<span class="badge badge-success">Success</span>'
                : '<span class="badge badge-danger">Failed</span>';

            item.innerHTML = `
                <div>
                    <div style="font-size: 0.875rem; color: var(--text-primary);">
                        ${icon} ${activity.description}
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">
                        ${this.formatDate(activity.timestamp)}
                    </div>
                </div>
                ${badge}
            `;

            list.appendChild(item);
        });

        container.appendChild(list);
    }

    async loadSuccessRates() {
        try {
            const response = await fetch('/api/stats/success-rates');
            const result = await response.json();

            if (result.success) {
                this.renderSuccessRates(result.data);
            }
        } catch (error) {
            console.error('Failed to load success rates:', error);
        }
    }

    renderSuccessRates(data) {
        const container = document.getElementById('success-rates-container');

        // Calculate success rates from real data
        const chartData = [];

        // Text prompts success rate
        if (data.text && data.text.length > 0) {
            const textTotal = data.text.reduce((sum, cat) => sum + (cat.total_tests || 0), 0);
            const textSuccess = data.text.reduce((sum, cat) => sum + (cat.successful_tests || 0), 0);
            const textRate = textTotal > 0 ? (textSuccess / textTotal * 100) : 0;
            chartData.push({ label: 'Text', value: Math.round(textRate), color: '#3b82f6' });
        } else {
            chartData.push({ label: 'Text', value: 0, color: '#3b82f6' });
        }

        // Multimodal success rate
        if (data.multimodal && data.multimodal.length > 0) {
            const mmTotal = data.multimodal.reduce((sum, type) => sum + (type.total_tests || 0), 0);
            const mmSuccess = data.multimodal.reduce((sum, type) => sum + (type.successful_tests || 0), 0);
            const mmRate = mmTotal > 0 ? (mmSuccess / mmTotal * 100) : 0;
            chartData.push({ label: 'Multimodal', value: Math.round(mmRate), color: '#8b5cf6' });
        } else {
            chartData.push({ label: 'Multimodal', value: 0, color: '#8b5cf6' });
        }

        // CTF success rate
        if (data.ctf && data.ctf.length > 0) {
            const ctfTotal = data.ctf.reduce((sum, cat) => sum + (cat.total_challenges || 0), 0);
            const ctfSuccess = data.ctf.reduce((sum, cat) => sum + (cat.solved_challenges || 0), 0);
            const ctfRate = ctfTotal > 0 ? (ctfSuccess / ctfTotal * 100) : 0;
            chartData.push({ label: 'CTF', value: Math.round(ctfRate), color: '#10b981' });
        } else {
            chartData.push({ label: 'CTF', value: 0, color: '#10b981' });
        }

        container.appendChild(ChartComponent.createBarChart(chartData));
    }

    getActivityIcon(type) {
        const icons = {
            'text_test': '📝',
            'multimodal_test': '🎨',
            'ctf_test': '🎯',
            'security_scan': '🛡️',
            'system_scan': '🖥️'
        };
        return icons[type] || '⚡';
    }

    formatDate(dateString) {
        if (!dateString) return '-';
        const date = new Date(dateString);
        return date.toLocaleString('ko-KR', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    async loadClassificationStats() {
        try {
            const response = await fetch('/api/stats/classification');
            const result = await response.json();

            if (result.success) {
                this.renderClassificationCharts(result.data);
            }
        } catch (error) {
            console.error('Failed to load classification stats:', error);
        }
    }

    renderClassificationCharts(data) {
        // Purpose Chart (Doughnut)
        this.renderPurposeChart(data.purpose);

        // Risk Category Chart (Doughnut)
        this.renderRiskChart(data.risk_category);

        // Technique Chart (Horizontal Bar)
        this.renderTechniqueChart(data.technique);

        // Success Rate Charts
        this.renderPurposeSuccessChart(data.purpose);
        this.renderRiskSuccessChart(data.risk_category);
    }

    renderPurposeChart(purposeData) {
        const ctx = document.getElementById('purposeChart');
        if (!ctx) return;

        const labels = purposeData.map(p => p.purpose);
        const values = purposeData.map(p => p.prompt_count);
        const colors = {
            'offensive': 'rgba(239, 68, 68, 0.8)',  // Red
            'defensive': 'rgba(34, 197, 94, 0.8)'    // Green
        };

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels.map(l => l === 'offensive' ? '⚔️ Offensive' : '🛡️ Defensive'),
                datasets: [{
                    data: values,
                    backgroundColor: labels.map(l => colors[l] || 'rgba(156, 163, 175, 0.8)'),
                    borderWidth: 2,
                    borderColor: '#1f2937'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#e5e7eb',
                            font: { size: 12 },
                            padding: 10
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percent = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value.toLocaleString()} (${percent}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    renderRiskChart(riskData) {
        const ctx = document.getElementById('riskChart');
        if (!ctx) return;

        const labels = riskData.map(r => r.risk_category);
        const values = riskData.map(r => r.prompt_count);
        const colors = {
            'security': 'rgba(59, 130, 246, 0.8)',     // Blue
            'safety': 'rgba(251, 191, 36, 0.8)',       // Yellow
            'ethics': 'rgba(168, 85, 247, 0.8)',       // Purple
            'compliance': 'rgba(34, 197, 94, 0.8)',    // Green
            'misinformation': 'rgba(239, 68, 68, 0.8)' // Red
        };

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels.map(l => {
                    const icons = {
                        'security': '🔒 Security',
                        'safety': '⚠️ Safety',
                        'ethics': '🎭 Ethics',
                        'compliance': '📋 Compliance',
                        'misinformation': '📰 Misinfo'
                    };
                    return icons[l] || l;
                }),
                datasets: [{
                    data: values,
                    backgroundColor: labels.map(l => colors[l] || 'rgba(156, 163, 175, 0.8)'),
                    borderWidth: 2,
                    borderColor: '#1f2937'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#e5e7eb',
                            font: { size: 11 },
                            padding: 8
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percent = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value.toLocaleString()} (${percent}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    renderTechniqueChart(techniqueData) {
        const ctx = document.getElementById('techniqueChart');
        if (!ctx) return;

        const labels = techniqueData.map(t => t.technique);
        const values = techniqueData.map(t => t.prompt_count);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Prompts',
                    data: values,
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.parsed.x.toLocaleString()} prompts`
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#9ca3af' },
                        grid: { color: 'rgba(75, 85, 99, 0.3)' }
                    },
                    y: {
                        ticks: { color: '#e5e7eb', font: { size: 11 } },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    renderPurposeSuccessChart(purposeData) {
        const ctx = document.getElementById('purposeSuccessChart');
        if (!ctx) return;

        const labels = purposeData.map(p => p.purpose === 'offensive' ? '⚔️ Offensive' : '🛡️ Defensive');
        const successRates = purposeData.map(p => (p.success_rate || 0).toFixed(1));
        const testCounts = purposeData.map(p => p.test_count || 0);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Success Rate (%)',
                    data: successRates,
                    backgroundColor: labels.map(l =>
                        l.includes('Offensive') ? 'rgba(239, 68, 68, 0.8)' : 'rgba(34, 197, 94, 0.8)'
                    ),
                    borderWidth: 2,
                    borderColor: '#1f2937'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const idx = context.dataIndex;
                                return [
                                    `Success Rate: ${context.parsed.y}%`,
                                    `Total Tests: ${testCounts[idx].toLocaleString()}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#9ca3af', callback: (value) => value + '%' },
                        grid: { color: 'rgba(75, 85, 99, 0.3)' }
                    },
                    x: {
                        ticks: { color: '#e5e7eb' },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    renderRiskSuccessChart(riskData) {
        const ctx = document.getElementById('riskSuccessChart');
        if (!ctx) return;

        const labels = riskData.map(r => {
            const icons = {
                'security': '🔒 Security',
                'safety': '⚠️ Safety',
                'ethics': '🎭 Ethics',
                'compliance': '📋 Compliance',
                'misinformation': '📰 Misinfo'
            };
            return icons[r.risk_category] || r.risk_category;
        });
        const successRates = riskData.map(r => (r.success_rate || 0).toFixed(1));
        const testCounts = riskData.map(r => r.test_count || 0);

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Success Rate (%)',
                    data: successRates,
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const idx = context.dataIndex;
                                return [
                                    `Success Rate: ${context.parsed.y}%`,
                                    `Total Tests: ${testCounts[idx].toLocaleString()}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#9ca3af', callback: (value) => value + '%' },
                        grid: { color: 'rgba(75, 85, 99, 0.3)' }
                    },
                    x: {
                        ticks: { color: '#e5e7eb', font: { size: 11 } },
                        grid: { display: false }
                    }
                }
            }
        });
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new HomePage();
});
