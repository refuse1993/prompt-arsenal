/**
 * Modal Component - shadcn style
 * 세련된 모달 다이얼로그
 */

export class Modal {
    constructor(options = {}) {
        this.title = options.title || '';
        this.size = options.size || 'default'; // default, large, full
        this.onClose = options.onClose || null;
        this.closeOnOutsideClick = options.closeOnOutsideClick !== false;
        this.element = null;
    }

    render(content) {
        this.element = document.createElement('div');
        this.element.className = 'modal-overlay';
        this.element.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(4px);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            animation: fadeIn 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        `;

        const maxWidth = this.size === 'large' ? '900px' : this.size === 'full' ? '95%' : '600px';

        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.cssText = `
            background: hsl(var(--card));
            border: 1px solid hsl(var(--border));
            border-radius: var(--radius);
            max-width: ${maxWidth};
            width: 90%;
            max-height: 85vh;
            overflow: auto;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
            animation: slideUp 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        `;

        modal.innerHTML = `
            <div class="modal-header" style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: var(--spacing-lg);
                border-bottom: 1px solid hsl(var(--border));
            ">
                <h3 style="
                    margin: 0;
                    color: hsl(var(--foreground));
                    font-size: 1.125rem;
                    font-weight: 600;
                    letter-spacing: -0.01em;
                ">${this.title}</h3>
                <button class="modal-close" style="
                    background: none;
                    border: 1px solid transparent;
                    color: hsl(var(--muted-foreground));
                    font-size: 1.25rem;
                    cursor: pointer;
                    padding: 0;
                    width: 32px;
                    height: 32px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: var(--radius);
                    transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
                " onmouseover="this.style.background='hsl(var(--accent))'; this.style.borderColor='hsl(var(--border))'" onmouseout="this.style.background='none'; this.style.borderColor='transparent'">×</button>
            </div>
            <div class="modal-body" style="padding: var(--spacing-lg);">
                ${typeof content === 'string' ? content : ''}
            </div>
        `;

        if (typeof content !== 'string') {
            modal.querySelector('.modal-body').appendChild(content);
        }

        // Close button
        modal.querySelector('.modal-close').addEventListener('click', () => this.close());

        // Close on outside click
        if (this.closeOnOutsideClick) {
            this.element.addEventListener('click', (e) => {
                if (e.target === this.element) {
                    this.close();
                }
            });
        }

        // ESC key
        this.handleEscape = (e) => {
            if (e.key === 'Escape') {
                this.close();
            }
        };
        document.addEventListener('keydown', this.handleEscape);

        this.element.appendChild(modal);

        return this.element;
    }

    show(content) {
        const rendered = this.render(content);
        document.body.appendChild(rendered);
        document.body.style.overflow = 'hidden';
    }

    close() {
        if (this.element) {
            this.element.style.animation = 'fadeOut 0.15s cubic-bezier(0.4, 0, 0.2, 1)';
            const modal = this.element.querySelector('.modal');
            modal.style.animation = 'slideDown 0.15s cubic-bezier(0.4, 0, 0.2, 1)';
            setTimeout(() => {
                this.element.remove();
                document.body.style.overflow = '';
                document.removeEventListener('keydown', this.handleEscape);
                if (this.onClose) {
                    this.onClose();
                }
            }, 150);
        }
    }

    static confirm(title, message, onConfirm) {
        const modal = new Modal({ title });

        const content = document.createElement('div');
        content.innerHTML = `
            <p style="
                color: hsl(var(--muted-foreground));
                margin-bottom: var(--spacing-lg);
                line-height: 1.6;
            ">${message}</p>
            <div style="display: flex; gap: var(--spacing-md); justify-content: flex-end;">
                <button class="btn btn-secondary cancel-btn">취소</button>
                <button class="btn btn-danger confirm-btn">확인</button>
            </div>
        `;

        content.querySelector('.cancel-btn').addEventListener('click', () => modal.close());
        content.querySelector('.confirm-btn').addEventListener('click', () => {
            onConfirm();
            modal.close();
        });

        modal.show(content);
    }

    static alert(title, message) {
        const modal = new Modal({ title });

        const content = document.createElement('div');
        content.innerHTML = `
            <p style="
                color: hsl(var(--muted-foreground));
                margin-bottom: var(--spacing-lg);
                line-height: 1.6;
            ">${message}</p>
            <div style="display: flex; justify-content: flex-end;">
                <button class="btn btn-primary close-btn">확인</button>
            </div>
        `;

        content.querySelector('.close-btn').addEventListener('click', () => modal.close());

        modal.show(content);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }

    @keyframes slideUp {
        from {
            transform: translateY(16px) scale(0.98);
            opacity: 0;
        }
        to {
            transform: translateY(0) scale(1);
            opacity: 1;
        }
    }

    @keyframes slideDown {
        from {
            transform: translateY(0) scale(1);
            opacity: 1;
        }
        to {
            transform: translateY(16px) scale(0.98);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
