/**
 * Chart Component
 * Simple chart components using CSS
 */

export class Chart {
    static createBarChart(data, options = {}) {
        const container = document.createElement('div');
        container.className = 'chart-container';
        container.style.cssText = 'display: flex; align-items: flex-end; gap: 0.5rem; height: 200px; padding: 1rem;';

        const maxValue = Math.max(...data.map(d => d.value));

        data.forEach(item => {
            const bar = document.createElement('div');
            const height = (item.value / maxValue) * 100;

            bar.style.cssText = `
                flex: 1;
                background: ${item.color || 'var(--primary)'};
                height: ${height}%;
                border-radius: 4px 4px 0 0;
                position: relative;
                transition: all 0.3s ease;
                cursor: pointer;
            `;

            bar.innerHTML = `
                <div style="position: absolute; top: -1.5rem; left: 50%; transform: translateX(-50%); font-size: 0.875rem; font-weight: 600; color: var(--text-primary);">
                    ${item.value}
                </div>
                <div style="position: absolute; bottom: -2rem; left: 50%; transform: translateX(-50%); font-size: 0.75rem; color: var(--text-secondary); white-space: nowrap;">
                    ${item.label}
                </div>
            `;

            bar.addEventListener('mouseenter', () => {
                bar.style.opacity = '0.8';
            });

            bar.addEventListener('mouseleave', () => {
                bar.style.opacity = '1';
            });

            container.appendChild(bar);
        });

        return container;
    }

    static createPieChart(data, options = {}) {
        const container = document.createElement('div');
        container.className = 'pie-chart';
        container.style.cssText = 'display: flex; flex-direction: column; gap: 1rem;';

        const total = data.reduce((sum, item) => sum + item.value, 0);

        // Visual representation
        const visual = document.createElement('div');
        visual.style.cssText = 'display: flex; height: 20px; border-radius: 10px; overflow: hidden;';

        data.forEach(item => {
            const segment = document.createElement('div');
            const percentage = (item.value / total) * 100;

            segment.style.cssText = `
                width: ${percentage}%;
                background: ${item.color || 'var(--primary)'};
                transition: all 0.3s ease;
            `;

            segment.title = `${item.label}: ${percentage.toFixed(1)}%`;
            visual.appendChild(segment);
        });

        container.appendChild(visual);

        // Legend
        const legend = document.createElement('div');
        legend.style.cssText = 'display: flex; flex-direction: column; gap: 0.5rem;';

        data.forEach(item => {
            const legendItem = document.createElement('div');
            const percentage = (item.value / total) * 100;

            legendItem.style.cssText = 'display: flex; justify-content: space-between; align-items: center;';
            legendItem.innerHTML = `
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: ${item.color || 'var(--primary)'};"></div>
                    <span style="font-size: 0.875rem; color: var(--text-secondary);">${item.label}</span>
                </div>
                <span style="font-weight: 600; color: var(--text-primary);">${percentage.toFixed(1)}%</span>
            `;

            legend.appendChild(legendItem);
        });

        container.appendChild(legend);

        return container;
    }

    static createLineChart(data, options = {}) {
        const container = document.createElement('div');
        container.style.cssText = 'position: relative; height: 200px; padding: 1rem;';

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.style.cssText = 'overflow: visible;';

        const maxValue = Math.max(...data.map(d => d.value));
        const points = data.map((item, index) => {
            const x = (index / (data.length - 1)) * 100;
            const y = 100 - (item.value / maxValue) * 100;
            return `${x},${y}`;
        }).join(' ');

        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', points);
        polyline.setAttribute('fill', 'none');
        polyline.setAttribute('stroke', options.color || 'var(--primary)');
        polyline.setAttribute('stroke-width', '2');
        polyline.setAttribute('vector-effect', 'non-scaling-stroke');

        svg.appendChild(polyline);

        // Add points
        data.forEach((item, index) => {
            const x = (index / (data.length - 1)) * 100;
            const y = 100 - (item.value / maxValue) * 100;

            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', `${x}%`);
            circle.setAttribute('cy', `${y}%`);
            circle.setAttribute('r', '4');
            circle.setAttribute('fill', options.color || 'var(--primary)');

            svg.appendChild(circle);
        });

        container.appendChild(svg);

        return container;
    }

    static createProgressBar(value, max = 100, options = {}) {
        const container = document.createElement('div');
        container.style.cssText = 'display: flex; flex-direction: column; gap: 0.5rem;';

        if (options.label) {
            const label = document.createElement('div');
            label.style.cssText = 'display: flex; justify-content: space-between; font-size: 0.875rem;';
            label.innerHTML = `
                <span style="color: var(--text-secondary);">${options.label}</span>
                <span style="font-weight: 600; color: var(--text-primary);">${value}/${max}</span>
            `;
            container.appendChild(label);
        }

        const bar = document.createElement('div');
        bar.style.cssText = 'height: 8px; background: var(--bg-hover); border-radius: 4px; overflow: hidden;';

        const fill = document.createElement('div');
        const percentage = (value / max) * 100;
        fill.style.cssText = `
            height: 100%;
            width: ${percentage}%;
            background: ${options.color || 'var(--primary)'};
            transition: width 0.3s ease;
        `;

        bar.appendChild(fill);
        container.appendChild(bar);

        return container;
    }
}
