/**
 * UI Extensions for Prompt Arsenal Dashboard
 * Additional features: Testing, Editing, Profiles, Variants
 */

// Global state
let currentProfiles = {};
let currentPromptForEdit = null;
let testingInProgress = false;

// ============================================================================
// Modal Management
// ============================================================================

function openAddPromptModal() {
    currentPromptForEdit = null;
    document.getElementById('prompt-modal-title').textContent = 'Add New Prompt';
    document.getElementById('prompt-form-category').value = '';
    document.getElementById('prompt-form-payload').value = '';
    document.getElementById('prompt-form-description').value = '';
    document.getElementById('prompt-form-tags').value = '';
    document.getElementById('prompt-modal').classList.remove('hidden');
    lucide.createIcons();
}

function openEditPromptModal(promptId) {
    console.log('openEditPromptModal called with ID:', promptId);
    console.log('window.promptsData:', window.promptsData);

    const prompt = window.promptsData.find(p => p.id === promptId);
    if (!prompt) {
        console.error('Prompt not found:', promptId);
        alert('Prompt not found!');
        return;
    }

    currentPromptForEdit = prompt;
    document.getElementById('prompt-modal-title').textContent = 'Edit Prompt';
    document.getElementById('prompt-form-category').value = prompt.category;
    document.getElementById('prompt-form-payload').value = prompt.payload;
    document.getElementById('prompt-form-description').value = prompt.description || '';
    document.getElementById('prompt-form-tags').value = prompt.tags || '';
    document.getElementById('prompt-modal').classList.remove('hidden');
    lucide.createIcons();
}

function closePromptModal() {
    document.getElementById('prompt-modal').classList.add('hidden');
    currentPromptForEdit = null;
}

async function savePrompt() {
    const category = document.getElementById('prompt-form-category').value;
    const payload = document.getElementById('prompt-form-payload').value;
    const description = document.getElementById('prompt-form-description').value;
    const tags = document.getElementById('prompt-form-tags').value;

    if (!category || !payload) {
        alert('Category and Payload are required!');
        return;
    }

    try {
        const url = currentPromptForEdit
            ? `${API_URL}/prompts/${currentPromptForEdit.id}`
            : `${API_URL}/prompts`;

        const method = currentPromptForEdit ? 'PUT' : 'POST';

        const res = await fetch(url, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category, payload, description, tags })
        });

        const data = await res.json();

        if (data.success) {
            alert(currentPromptForEdit ? 'Prompt updated!' : 'Prompt created!');
            closePromptModal();
            await loadData();  // Reload prompts
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Save error:', error);
        alert('Failed to save prompt: ' + error.message);
    }
}

async function deletePrompt(promptId) {
    if (!confirm('Are you sure you want to delete this prompt?')) {
        return;
    }

    try {
        const res = await fetch(`${API_URL}/prompts/${promptId}`, {
            method: 'DELETE'
        });

        const data = await res.json();

        if (data.success) {
            alert('Prompt deleted!');
            await loadData();  // Reload prompts
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert('Failed to delete prompt: ' + error.message);
    }
}

// ============================================================================
// Testing
// ============================================================================

async function openTestModal(promptId) {
    console.log('openTestModal called with ID:', promptId);

    const prompt = window.promptsData.find(p => p.id === promptId);
    if (!prompt) {
        console.error('Prompt not found:', promptId);
        alert('Prompt not found!');
        return;
    }

    document.getElementById('test-modal').classList.remove('hidden');
    document.getElementById('test-prompt-preview').textContent = prompt.payload.substring(0, 200) + '...';
    document.getElementById('test-prompt-id').value = prompt.id;
    document.getElementById('test-prompt-text').value = prompt.payload;

    // Load profiles
    await loadProfilesForTest();
    lucide.createIcons();
}

function closeTestModal() {
    document.getElementById('test-modal').classList.add('hidden');
}

async function loadProfilesForTest() {
    try {
        const res = await fetch(`${API_URL}/profiles`);
        const data = await res.json();

        if (data.success) {
            currentProfiles = data.data.profiles;
            const select = document.getElementById('test-profile-select');
            select.innerHTML = '<option value="">-- Select Profile --</option>';

            Object.keys(currentProfiles).forEach(name => {
                const profile = currentProfiles[name];
                const option = document.createElement('option');
                option.value = name;
                option.textContent = `${name} (${profile.provider}/${profile.model})`;
                select.appendChild(option);
            });

            if (data.data.default_profile) {
                select.value = data.data.default_profile;
            }
        }
    } catch (error) {
        console.error('Load profiles error:', error);
    }
}

async function runTest() {
    const profileName = document.getElementById('test-profile-select').value;
    const promptId = document.getElementById('test-prompt-id').value;
    const promptText = document.getElementById('test-prompt-text').value;

    if (!profileName) {
        alert('Please select a profile!');
        return;
    }

    if (!promptText) {
        alert('Prompt text is empty!');
        return;
    }

    const profile = currentProfiles[profileName];

    testingInProgress = true;
    document.getElementById('test-run-btn').disabled = true;
    document.getElementById('test-result-container').innerHTML = `
        <div class="p-8 text-center">
            <div class="inline-block animate-spin rounded-full h-12 w-12 border-4 border-gray-700 border-t-blue-500"></div>
            <p class="mt-4 text-gray-400">Testing with ${profile.provider}/${profile.model}...</p>
        </div>
    `;

    try {
        const res = await fetch(`${API_URL}/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt_id: promptId || null,
                prompt_text: promptText,
                provider: profile.provider,
                model: profile.model,
                api_key: profile.api_key
            })
        });

        const data = await res.json();

        if (data.success) {
            const result = data.data;
            const successClass = result.success ? 'bg-green-900 border-green-700 text-green-300' : 'bg-red-900 border-red-700 text-red-300';

            document.getElementById('test-result-container').innerHTML = `
                <div class="card p-6 ${successClass}">
                    <h3 class="font-semibold text-lg mb-2">
                        ${result.success ? '✅ SUCCESS' : '❌ FAILED'}
                    </h3>
                    <div class="mb-4">
                        <p class="text-sm mb-1">Severity: <span class="font-semibold">${result.severity || 'N/A'}</span></p>
                        <p class="text-sm mb-1">Confidence: <span class="font-semibold">${((result.confidence || 0) * 100).toFixed(0)}%</span></p>
                        <p class="text-sm mb-1">Response Time: <span class="font-semibold">${(result.response_time || 0).toFixed(2)}s</span></p>
                    </div>
                    <div class="mb-4">
                        <p class="text-xs font-semibold uppercase text-gray-400 mb-2">Response:</p>
                        <p class="text-sm bg-black bg-opacity-30 p-3 rounded">${escapeHtml(result.response || 'No response')}</p>
                    </div>
                    ${result.reasoning ? `
                        <div>
                            <p class="text-xs font-semibold uppercase text-gray-400 mb-2">Reasoning:</p>
                            <p class="text-sm italic">${escapeHtml(result.reasoning)}</p>
                        </div>
                    ` : ''}
                </div>
            `;

            // Reload data to update test counts
            if (promptId) {
                setTimeout(() => loadData(), 1000);
            }
        } else {
            document.getElementById('test-result-container').innerHTML = `
                <div class="card p-6 bg-red-900 border-red-700">
                    <h3 class="font-semibold text-lg mb-2">❌ Error</h3>
                    <p class="text-sm">${data.error}</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Test error:', error);
        document.getElementById('test-result-container').innerHTML = `
            <div class="card p-6 bg-red-900 border-red-700">
                <h3 class="font-semibold text-lg mb-2">❌ Error</h3>
                <p class="text-sm">${error.message}</p>
            </div>
        `;
    } finally {
        testingInProgress = false;
        document.getElementById('test-run-btn').disabled = false;
    }
}

// ============================================================================
// Settings (API Profiles)
// ============================================================================

async function openSettingsModal() {
    document.getElementById('settings-modal').classList.remove('hidden');
    await loadProfilesList();
    lucide.createIcons();
}

function closeSettingsModal() {
    document.getElementById('settings-modal').classList.add('hidden');
}

async function loadProfilesList() {
    try {
        const res = await fetch(`${API_URL}/profiles`);
        const data = await res.json();

        if (data.success) {
            const container = document.getElementById('profiles-list');
            const profiles = data.data.profiles;

            if (Object.keys(profiles).length === 0) {
                container.innerHTML = '<p class="text-gray-500 text-center py-8">No profiles yet. Create one below!</p>';
                return;
            }

            container.innerHTML = Object.keys(profiles).map(name => {
                const profile = profiles[name];
                const isDefault = name === data.data.default_profile;
                return `
                    <div class="card p-4 flex justify-between items-center">
                        <div>
                            <p class="font-semibold text-white flex items-center gap-2">
                                ${name}
                                ${isDefault ? '<span class="badge badge-success text-xs">Default</span>' : ''}
                            </p>
                            <p class="text-sm text-gray-400">${profile.provider} / ${profile.model}</p>
                            ${profile.base_url ? `<p class="text-xs text-gray-500">${profile.base_url}</p>` : ''}
                        </div>
                        <button onclick="deleteProfile('${name}')" class="btn btn-danger btn-sm">
                            Delete
                        </button>
                    </div>
                `;
            }).join('');
        }
    } catch (error) {
        console.error('Load profiles error:', error);
    }
}

async function createProfile() {
    const name = document.getElementById('profile-name').value;
    const provider = document.getElementById('profile-provider').value;
    const model = document.getElementById('profile-model').value;
    const apiKey = document.getElementById('profile-api-key').value;
    const baseUrl = document.getElementById('profile-base-url').value;

    if (!name || !provider || !model) {
        alert('Name, Provider, and Model are required!');
        return;
    }

    try {
        const res = await fetch(`${API_URL}/profiles`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, provider, model, api_key: apiKey, base_url: baseUrl || null })
        });

        const data = await res.json();

        if (data.success) {
            alert('Profile created!');
            // Clear form
            document.getElementById('profile-name').value = '';
            document.getElementById('profile-provider').value = '';
            document.getElementById('profile-model').value = '';
            document.getElementById('profile-api-key').value = '';
            document.getElementById('profile-base-url').value = '';
            await loadProfilesList();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Create profile error:', error);
        alert('Failed to create profile: ' + error.message);
    }
}

async function deleteProfile(profileName) {
    if (!confirm(`Delete profile "${profileName}"?`)) {
        return;
    }

    try {
        const res = await fetch(`${API_URL}/profiles/${profileName}`, {
            method: 'DELETE'
        });

        const data = await res.json();

        if (data.success) {
            alert('Profile deleted!');
            await loadProfilesList();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Delete profile error:', error);
        alert('Failed to delete profile: ' + error.message);
    }
}

// ============================================================================
// Variants Generation
// ============================================================================

async function openVariantsModal(promptId) {
    console.log('openVariantsModal called with ID:', promptId);

    const prompt = window.promptsData.find(p => p.id === promptId);
    if (!prompt) {
        console.error('Prompt not found:', promptId);
        alert('Prompt not found!');
        return;
    }

    document.getElementById('variants-modal').classList.remove('hidden');
    document.getElementById('variants-base-payload').value = prompt.payload;
    document.getElementById('variants-category').value = prompt.category;
    lucide.createIcons();
}

function closeVariantsModal() {
    document.getElementById('variants-modal').classList.add('hidden');
}

async function generateVariants() {
    const basePayload = document.getElementById('variants-base-payload').value;
    const strategies = Array.from(document.querySelectorAll('input[name="variant-strategy"]:checked')).map(cb => cb.value);

    if (!basePayload) {
        alert('Base payload is empty!');
        return;
    }

    if (strategies.length === 0) {
        alert('Select at least one strategy!');
        return;
    }

    document.getElementById('variants-result-container').innerHTML = `
        <div class="p-8 text-center">
            <div class="inline-block animate-spin rounded-full h-12 w-12 border-4 border-gray-700 border-t-blue-500"></div>
            <p class="mt-4 text-gray-400">Generating variants...</p>
        </div>
    `;

    try {
        const res = await fetch(`${API_URL}/generate-variants`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ payload: basePayload, strategies })
        });

        const data = await res.json();

        if (data.success) {
            const variants = data.data.variants;
            document.getElementById('variants-result-container').innerHTML = `
                <div class="space-y-3">
                    <p class="text-sm text-gray-400 mb-4">Generated ${variants.length} variants</p>
                    ${variants.map(v => `
                        <div class="card p-4">
                            <div class="flex justify-between items-start mb-2">
                                <span class="badge badge-info text-xs">${v.method}</span>
                                <button onclick="saveVariantAsPrompt('${escapeHtml(v.payload)}', '${v.method}')" class="btn btn-sm btn-success">
                                    Save as Prompt
                                </button>
                            </div>
                            <pre class="mono text-sm text-gray-300 bg-black bg-opacity-30 p-3 rounded overflow-x-auto">${escapeHtml(v.payload)}</pre>
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            document.getElementById('variants-result-container').innerHTML = `
                <div class="card p-6 bg-red-900 border-red-700">
                    <p class="text-sm">${data.error}</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Generate variants error:', error);
        document.getElementById('variants-result-container').innerHTML = `
            <div class="card p-6 bg-red-900 border-red-700">
                <p class="text-sm">${error.message}</p>
            </div>
        `;
    }
}

async function saveVariantAsPrompt(payload, method) {
    const category = document.getElementById('variants-category').value;
    const description = `Generated using ${method}`;

    try {
        const res = await fetch(`${API_URL}/prompts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                category,
                payload,
                description,
                source: 'variant-generator',
                tags: method
            })
        });

        const data = await res.json();

        if (data.success) {
            alert('Variant saved as new prompt!');
            await loadData();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Save variant error:', error);
        alert('Failed to save variant: ' + error.message);
    }
}
