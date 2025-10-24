/**
 * Home Page Module
 * Dashboard overview and statistics
 */

import { Card } from '../components/Card.js';
import { Chart } from '../components/Chart.js';

class HomePage {
    constructor() {
        this.loadStats();
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

        container.appendChild(Chart.createBarChart(chartData));
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
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new HomePage();
});
