/**
 * Security Page Module
 * Security scanner management
 */

import { Card } from '../components/Card.js';
import { Table, createBadge, formatDate } from '../components/Table.js';
import { Pagination } from '../components/Pagination.js';
import { Modal } from '../components/Modal.js';

class SecurityPage {
    constructor() {
        this.currentPage = 1;
        this.currentSeverity = null;
        this.pagination = new Pagination({
            currentPage: 1,
            totalPages: 1,
            onPageChange: (page) => this.loadScanResults(page)
        });

        this.init();
    }

    async init() {
        await this.loadStats();
        await this.loadTools();
        this.renderFilters();
        await this.loadVulnerabilities();
        await this.loadScanResults();
    }

    async loadStats() {
        try {
            const response = await fetch('/api/security/stats');
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
            stats.total_scans,
            'Total Scans',
            '🔍'
        ));

        container.appendChild(Card.createStatCard(
            stats.critical_vulns,
            'Critical',
            '🚨'
        ));

        container.appendChild(Card.createStatCard(
            stats.high_vulns,
            'High',
            '⚠️'
        ));

        container.appendChild(Card.createStatCard(
            stats.medium_vulns + stats.low_vulns,
            'Medium/Low',
            'ℹ️'
        ));
    }

    async loadTools() {
        try {
            const response = await fetch('/api/security/tools');
            const result = await response.json();

            if (result.success) {
                this.renderTools(result.data);
            }
        } catch (error) {
            console.error('Failed to load tools:', error);
        }
    }

    renderTools(tools) {
        const container = document.getElementById('tools-container');

        const grid = document.createElement('div');
        grid.className = 'grid grid-3';

        tools.forEach(tool => {
            const items = [
                { label: 'Version', value: tool.version },
                { label: 'Status', value: tool.enabled ? '✅ Enabled' : '❌ Disabled' }
            ];

            const card = Card.createInfoCard(tool.name, items, '🔧');

            const description = document.createElement('p');
            description.style.cssText = 'font-size: 0.875rem; color: var(--text-secondary); margin-top: 0.5rem;';
            description.textContent = tool.description;

            card.querySelector('.card-body').appendChild(description);

            grid.appendChild(card);
        });

        container.appendChild(grid);
    }

    renderFilters() {
        const container = document.getElementById('filter-container');

        container.innerHTML = `
            <select id="severity-filter" style="
                padding: var(--spacing-sm) var(--spacing-md);
                background: var(--bg-hover);
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                color: var(--text-primary);
            ">
                <option value="">All Severities</option>
                <option value="critical">🚨 Critical</option>
                <option value="high">⚠️ High</option>
                <option value="medium">ℹ️ Medium</option>
                <option value="low">✅ Low</option>
            </select>
        `;

        document.getElementById('severity-filter').addEventListener('change', (e) => {
            this.currentSeverity = e.target.value || null;
            this.loadVulnerabilities();
        });
    }

    async loadVulnerabilities() {
        try {
            let url = '/api/security/vulnerabilities';
            if (this.currentSeverity) {
                url += `?severity=${this.currentSeverity}`;
            }

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.renderVulnerabilities(result.data);
            }
        } catch (error) {
            console.error('Failed to load vulnerabilities:', error);
        }
    }

    renderVulnerabilities(vulnerabilities) {
        const container = document.getElementById('vulnerabilities-container');

        if (vulnerabilities.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No vulnerabilities found.</p>';
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
                    key: 'severity',
                    label: 'Severity',
                    sortable: true,
                    render: (value) => {
                        const badges = {
                            'critical': createBadge('🚨 Critical', 'danger'),
                            'high': createBadge('⚠️ High', 'warning'),
                            'medium': createBadge('ℹ️ Medium', 'info'),
                            'low': createBadge('✅ Low', 'success')
                        };
                        return badges[value] || value;
                    }
                },
                {
                    key: 'type',
                    label: 'Type',
                    sortable: true
                },
                {
                    key: 'description',
                    label: 'Description',
                    sortable: false,
                    render: (value) => {
                        const truncated = value.length > 80 ? value.substring(0, 80) + '...' : value;
                        return `<span style="font-size: 0.875rem;">${truncated}</span>`;
                    }
                },
                {
                    key: 'file',
                    label: 'File',
                    sortable: true,
                    render: (value) => {
                        const filename = value.split('/').pop();
                        return `<code style="font-size: 0.875rem;">${filename}</code>`;
                    }
                },
                {
                    key: 'line',
                    label: 'Line',
                    sortable: true
                }
            ],
            data: vulnerabilities,
            onRowClick: (row) => this.showVulnerabilityDetails(row)
        });

        container.innerHTML = '';
        container.appendChild(table.render());
    }

    async loadScanResults(page = 1) {
        this.currentPage = page;

        try {
            const response = await fetch(`/api/security/scan-results?page=${page}&limit=50`);
            const result = await response.json();

            if (result.success) {
                this.renderScanResults(result.data);

                if (result.pagination) {
                    this.pagination.currentPage = result.pagination.page;
                    this.pagination.totalPages = result.pagination.pages;
                    this.renderPagination();
                }
            }
        } catch (error) {
            console.error('Failed to load scan results:', error);
        }
    }

    renderScanResults(results) {
        const container = document.getElementById('scan-results-container');

        if (results.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No scan results yet. Run your first scan!</p>';
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
                    key: 'tool',
                    label: 'Tool',
                    sortable: true,
                    render: (value) => createBadge(value, 'primary')
                },
                {
                    key: 'target',
                    label: 'Target',
                    sortable: false
                },
                {
                    key: 'vulnerabilities_found',
                    label: 'Vulnerabilities',
                    sortable: true
                },
                {
                    key: 'status',
                    label: 'Status',
                    sortable: true,
                    render: (value) => {
                        const badges = {
                            'completed': createBadge('Completed', 'success'),
                            'running': createBadge('Running', 'warning'),
                            'failed': createBadge('Failed', 'danger')
                        };
                        return badges[value] || value;
                    }
                },
                {
                    key: 'scanned_at',
                    label: 'Scanned',
                    sortable: true,
                    render: formatDate
                }
            ],
            data: results,
            onRowClick: (row) => this.showScanDetails(row)
        });

        container.innerHTML = '';
        container.appendChild(table.render());
    }

    renderPagination() {
        const container = document.getElementById('pagination-container');
        container.innerHTML = '';
        container.appendChild(this.pagination.render());
    }

    showVulnerabilityDetails(vuln) {
        const modal = new Modal({ title: `Vulnerability #${vuln.id} - ${vuln.type}`, size: 'large' });

        const severityBadges = {
            'critical': createBadge('🚨 Critical', 'danger'),
            'high': createBadge('⚠️ High', 'warning'),
            'medium': createBadge('ℹ️ Medium', 'info'),
            'low': createBadge('✅ Low', 'success')
        };

        const severityColor = {
            'critical': 'hsl(var(--destructive))',
            'high': 'hsl(var(--warning))',
            'medium': 'hsl(142.1 70.6% 45.3%)',
            'low': 'hsl(var(--success))'
        };

        const content = document.createElement('div');
        content.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Severity</div>
                    <div>${severityBadges[vuln.severity]}</div>
                </div>
                ${vuln.cwe_id ? `
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">CWE ID</div>
                    <div style="font-family: monospace;">${vuln.cwe_id}</div>
                </div>
                ` : ''}
                ${vuln.confidence ? `
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Confidence</div>
                    <div style="font-weight: 600;">${Math.round(vuln.confidence * 100)}%</div>
                </div>
                ` : ''}
            </div>

            <div style="margin-bottom: 1.5rem;">
                <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Type</div>
                <div style="
                    padding: 0.75rem;
                    background: hsl(var(--card));
                    border: 1px solid hsl(var(--border));
                    border-radius: var(--radius-sm);
                    font-weight: 600;
                ">${vuln.type}</div>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Description</div>
                <div style="
                    padding: 1rem;
                    background: hsl(var(--muted) / 0.2);
                    border-left: 3px solid ${severityColor[vuln.severity] || 'hsl(var(--border))'};
                    border-radius: var(--radius-sm);
                    line-height: 1.6;
                ">${vuln.description || 'No description available'}</div>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Location</div>
                <div style="
                    padding: 0.75rem;
                    background: hsl(var(--card));
                    border: 1px solid hsl(var(--border));
                    border-radius: var(--radius-sm);
                ">
                    <div style="font-family: monospace; font-size: 0.875rem; margin-bottom: 0.5rem; word-break: break-all;">
                        📄 ${vuln.file}
                    </div>
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground));">
                        Line ${vuln.line}
                    </div>
                </div>
            </div>

            ${vuln.recommendation ? `
                <div style="margin-bottom: 1.5rem;">
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">💡 Remediation</div>
                    <div style="
                        padding: 1rem;
                        background: hsl(var(--success) / 0.1);
                        border-left: 3px solid hsl(var(--success));
                        border-radius: var(--radius-sm);
                        line-height: 1.6;
                    ">${vuln.recommendation}</div>
                </div>
            ` : ''}

            ${vuln.verified_by ? `
                <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem; background: hsl(var(--muted) / 0.2); border-radius: var(--radius-sm);">
                    <span style="font-size: 0.875rem; color: hsl(var(--muted-foreground));">Verified by:</span>
                    <span style="font-weight: 600;">${vuln.verified_by}</span>
                    ${vuln.is_false_positive ? '<span style="margin-left: auto; color: hsl(var(--warning));">⚠️ Flagged as False Positive</span>' : ''}
                </div>
            ` : ''}
        `;

        modal.show(content);
    }

    showScanDetails(scan) {
        const modal = new Modal({ title: `Scan #${scan.id} - ${scan.tool}`, size: 'large' });

        const content = document.createElement('div');

        const statusBadge = scan.status === 'completed' ? createBadge('Completed', 'success') :
                           scan.status === 'running' ? createBadge('Running', 'warning') :
                           createBadge('Failed', 'danger');

        content.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div style="
                    padding: 1.25rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-md);
                    text-align: center;
                ">
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem; font-weight: 600;">Total</div>
                    <div style="font-size: 2rem; font-weight: 700; color: hsl(var(--primary));">${scan.vulnerabilities_found}</div>
                </div>
                <div style="
                    padding: 1.25rem;
                    background: hsl(var(--destructive) / 0.1);
                    border-radius: var(--radius-md);
                    text-align: center;
                ">
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem; font-weight: 600;">Critical</div>
                    <div style="font-size: 2rem; font-weight: 700; color: hsl(var(--destructive));">${scan.critical_count || 0}</div>
                </div>
                <div style="
                    padding: 1.25rem;
                    background: hsl(var(--warning) / 0.1);
                    border-radius: var(--radius-md);
                    text-align: center;
                ">
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem; font-weight: 600;">High</div>
                    <div style="font-size: 2rem; font-weight: 700; color: hsl(var(--warning));">${scan.high_count || 0}</div>
                </div>
                <div style="
                    padding: 1.25rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-md);
                    text-align: center;
                ">
                    <div style="font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem; font-weight: 600;">Medium/Low</div>
                    <div style="font-size: 2rem; font-weight: 700; color: hsl(var(--foreground));">${(scan.medium_count || 0) + (scan.low_count || 0)}</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Tool</div>
                    <div>${createBadge(scan.tool, 'primary')}</div>
                </div>
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Status</div>
                    <div>${statusBadge}</div>
                </div>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Target Path</div>
                <code style="
                    display: block;
                    padding: 0.75rem;
                    background: hsl(var(--muted) / 0.2);
                    border-radius: var(--radius-sm);
                    font-size: 0.8125rem;
                    word-break: break-all;
                ">${scan.target}</code>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Duration</div>
                    <div style="font-size: 1.125rem; font-weight: 600;">${scan.scan_duration ? Math.round(scan.scan_duration) + 's' : 'N/A'}</div>
                </div>
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">LLM Calls</div>
                    <div style="font-size: 1.125rem; font-weight: 600;">${scan.llm_calls || 0}</div>
                </div>
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">LLM Cost</div>
                    <div style="font-size: 1.125rem; font-weight: 600;">$${(scan.llm_cost || 0).toFixed(4)}</div>
                </div>
            </div>

            <div style="margin-bottom: 0.5rem;">
                <div style="font-weight: 600; font-size: 0.875rem; color: hsl(var(--muted-foreground)); margin-bottom: 0.5rem;">Scanned At</div>
                <div>${formatDate(scan.scanned_at)}</div>
            </div>
        `;

        modal.show(content);
    }

    showRunScanModal() {
        const modal = new Modal({ title: 'Run Security Scan' });

        const content = document.createElement('div');
        content.innerHTML = `
            <form id="run-scan-form">
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Scanner Tool</label>
                    <select name="tool" required style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                        <option value="garak">Garak - LLM Vulnerability Scanner</option>
                        <option value="semgrep">Semgrep - Static Analysis</option>
                        <option value="bandit">Bandit - Python Security</option>
                    </select>
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Target Path</label>
                    <input type="text" name="target" required placeholder="/path/to/code" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                </div>
                <div class="flex gap-1 mt-2">
                    <button type="submit" class="btn btn-primary">▶️ Run Scan</button>
                    <button type="button" class="btn btn-secondary cancel-btn">Cancel</button>
                </div>
            </form>
        `;

        content.querySelector('.cancel-btn').addEventListener('click', () => modal.close());

        content.querySelector('#run-scan-form').addEventListener('submit', (e) => {
            e.preventDefault();
            modal.close();
            Modal.alert('Success', 'Security scan started! Results will appear in the scan history.');
            setTimeout(() => this.loadScanResults(1), 1000);
        });

        modal.show(content);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new SecurityPage();
});
