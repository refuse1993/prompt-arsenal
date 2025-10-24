/**
 * System Page Module
 * System scanner and network security
 */

import { Card } from '../components/Card.js';
import { Table, createBadge, formatDate } from '../components/Table.js';
import { Pagination } from '../components/Pagination.js';
import { Modal } from '../components/Modal.js';

class SystemPage {
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
        await this.loadHosts();
        await this.loadPorts();
        this.renderFilters();
        await this.loadCVEs();
        await this.loadScanResults();

        document.getElementById('run-scan-btn').addEventListener('click', () => {
            this.showRunScanModal();
        });
    }

    async loadStats() {
        try {
            const response = await fetch('/api/system/stats');
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
            stats.total_hosts,
            'Hosts Found',
            '🖥️'
        ));

        container.appendChild(Card.createStatCard(
            stats.total_ports,
            'Open Ports',
            '🔓'
        ));

        container.appendChild(Card.createStatCard(
            stats.critical_cves + stats.high_cves,
            'Critical CVEs',
            '🚨'
        ));
    }

    async loadTools() {
        try {
            const response = await fetch('/api/system/tools');
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
        grid.className = 'grid grid-2';

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

    async loadHosts() {
        try {
            const response = await fetch('/api/system/hosts');
            const result = await response.json();

            if (result.success) {
                this.renderHosts(result.data);
            }
        } catch (error) {
            console.error('Failed to load hosts:', error);
        }
    }

    renderHosts(hosts) {
        const container = document.getElementById('hosts-container');

        if (hosts.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No hosts scanned yet.</p>';
            return;
        }

        const list = document.createElement('div');
        list.style.cssText = 'display: flex; flex-direction: column; gap: var(--spacing-sm);';

        hosts.forEach(host => {
            const item = document.createElement('div');
            item.style.cssText = `
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: var(--spacing-sm) var(--spacing-md);
                background: var(--bg-hover);
                border-radius: var(--radius-md);
                cursor: pointer;
                transition: var(--transition);
            `;

            item.addEventListener('mouseenter', () => {
                item.style.background = 'var(--primary)';
            });

            item.addEventListener('mouseleave', () => {
                item.style.background = 'var(--bg-hover)';
            });

            item.innerHTML = `
                <div>
                    <div style="font-weight: 600;">${host.ip}</div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">${host.hostname || 'Unknown'}</div>
                </div>
                ${createBadge(`${host.open_ports} ports`, 'info')}
            `;

            list.appendChild(item);
        });

        container.appendChild(list);
    }

    async loadPorts() {
        try {
            const response = await fetch('/api/system/ports');
            const result = await response.json();

            if (result.success) {
                this.renderPorts(result.data);
            }
        } catch (error) {
            console.error('Failed to load ports:', error);
        }
    }

    renderPorts(ports) {
        const container = document.getElementById('ports-container');

        if (ports.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No open ports found.</p>';
            return;
        }

        const list = document.createElement('div');
        list.style.cssText = 'display: flex; flex-direction: column; gap: var(--spacing-sm);';

        ports.slice(0, 10).forEach(port => {
            const item = document.createElement('div');
            item.style.cssText = `
                display: flex;
                justify-content: space-between;
                padding: var(--spacing-sm) var(--spacing-md);
                background: var(--bg-hover);
                border-radius: var(--radius-md);
            `;

            item.innerHTML = `
                <div>
                    <div style="font-weight: 600;">Port ${port.port}</div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">${port.service || 'Unknown'}</div>
                </div>
                ${createBadge(port.state, 'success')}
            `;

            list.appendChild(item);
        });

        container.appendChild(list);
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
            this.loadCVEs();
        });
    }

    async loadCVEs() {
        try {
            let url = '/api/system/cves';
            if (this.currentSeverity) {
                url += `?severity=${this.currentSeverity}`;
            }

            const response = await fetch(url);
            const result = await response.json();

            if (result.success) {
                this.renderCVEs(result.data);
            }
        } catch (error) {
            console.error('Failed to load CVEs:', error);
        }
    }

    renderCVEs(cves) {
        const container = document.getElementById('cves-container');

        if (cves.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No CVEs found.</p>';
            return;
        }

        const table = new Table({
            columns: [
                {
                    key: 'cve_id',
                    label: 'CVE ID',
                    sortable: true,
                    render: (value) => `<a href="https://nvd.nist.gov/vuln/detail/${value}" target="_blank" style="color: var(--primary);">${value}</a>`
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
                    key: 'score',
                    label: 'CVSS Score',
                    sortable: true,
                    render: (value) => {
                        const color = value >= 9 ? 'danger' : value >= 7 ? 'warning' : 'info';
                        return createBadge(value.toFixed(1), color);
                    }
                },
                {
                    key: 'description',
                    label: 'Description',
                    sortable: false,
                    render: (value) => {
                        const truncated = value.length > 100 ? value.substring(0, 100) + '...' : value;
                        return `<span style="font-size: 0.875rem;">${truncated}</span>`;
                    }
                },
                {
                    key: 'affected_service',
                    label: 'Service',
                    sortable: true
                }
            ],
            data: cves,
            onRowClick: (row) => this.showCVEDetails(row)
        });

        container.innerHTML = '';
        container.appendChild(table.render());
    }

    async loadScanResults(page = 1) {
        this.currentPage = page;

        try {
            const response = await fetch(`/api/system/scan-results?page=${page}&limit=50`);
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
                    key: 'target',
                    label: 'Target',
                    sortable: false
                },
                {
                    key: 'hosts_found',
                    label: 'Hosts',
                    sortable: true
                },
                {
                    key: 'ports_found',
                    label: 'Ports',
                    sortable: true
                },
                {
                    key: 'cves_found',
                    label: 'CVEs',
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

    showCVEDetails(cve) {
        const modal = new Modal({ title: cve.cve_id });

        const severityBadges = {
            'critical': createBadge('🚨 Critical', 'danger'),
            'high': createBadge('⚠️ High', 'warning'),
            'medium': createBadge('ℹ️ Medium', 'info'),
            'low': createBadge('✅ Low', 'success')
        };

        const content = document.createElement('div');
        content.innerHTML = `
            <div class="mb-1">
                <strong>CVE ID:</strong> <a href="https://nvd.nist.gov/vuln/detail/${cve.cve_id}" target="_blank" style="color: var(--primary);">${cve.cve_id}</a>
            </div>
            <div class="mb-1">
                <strong>Severity:</strong> ${severityBadges[cve.severity]}
            </div>
            <div class="mb-1">
                <strong>CVSS Score:</strong> ${createBadge(cve.score.toFixed(1), 'warning')}
            </div>
            <div class="mb-1">
                <strong>Description:</strong>
                <p style="color: var(--text-secondary); margin-top: 0.5rem;">${cve.description}</p>
            </div>
            <div class="mb-1">
                <strong>Affected Service:</strong> ${cve.affected_service}
            </div>
            <div class="mb-1">
                <strong>Mitigation:</strong>
                <p style="color: var(--text-secondary); margin-top: 0.5rem;">${cve.mitigation || 'Update to latest version'}</p>
            </div>
        `;

        modal.show(content);
    }

    showScanDetails(scan) {
        const modal = new Modal({ title: `Scan #${scan.id}` });

        const content = document.createElement('div');
        content.innerHTML = `
            <div class="mb-1">
                <strong>Target:</strong> ${scan.target}
            </div>
            <div class="mb-1">
                <strong>Hosts Found:</strong> ${scan.hosts_found}
            </div>
            <div class="mb-1">
                <strong>Ports Found:</strong> ${scan.ports_found}
            </div>
            <div class="mb-1">
                <strong>CVEs Found:</strong> ${scan.cves_found}
            </div>
            <div class="mb-1">
                <strong>Status:</strong> ${scan.status}
            </div>
            <div class="mb-1">
                <strong>Scanned:</strong> ${formatDate(scan.scanned_at)}
            </div>
        `;

        modal.show(content);
    }

    showRunScanModal() {
        const modal = new Modal({ title: 'Run Network Scan' });

        const content = document.createElement('div');
        content.innerHTML = `
            <form id="run-scan-form">
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Target</label>
                    <input type="text" name="target" required placeholder="192.168.1.0/24 or example.com" style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                </div>
                <div class="mb-1">
                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-secondary);">Scan Type</label>
                    <select name="scan_type" required style="
                        width: 100%;
                        padding: var(--spacing-md);
                        background: var(--bg-hover);
                        border: 1px solid var(--border);
                        border-radius: var(--radius-md);
                        color: var(--text-primary);
                    ">
                        <option value="quick">Quick Scan (Top 100 ports)</option>
                        <option value="full">Full Scan (All ports)</option>
                        <option value="vuln">Vulnerability Scan (with CVE check)</option>
                    </select>
                </div>
                <div class="flex gap-1 mt-2">
                    <button type="submit" class="btn btn-primary">▶️ Start Scan</button>
                    <button type="button" class="btn btn-secondary cancel-btn">Cancel</button>
                </div>
            </form>
        `;

        content.querySelector('.cancel-btn').addEventListener('click', () => modal.close());

        content.querySelector('#run-scan-form').addEventListener('submit', (e) => {
            e.preventDefault();
            modal.close();
            Modal.alert('Success', 'Network scan started! Results will appear in the scan history.');
            setTimeout(() => this.loadScanResults(1), 1000);
        });

        modal.show(content);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    new SystemPage();
});
