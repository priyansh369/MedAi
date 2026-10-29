/**
 * app.js
 * ------
 * Client-side Controller for Medical AI Report Assistant & NLP Classifier.
 * Handles:
 * 1. Tab switching between Report Analyzer, Abstract Classifier, Performance, Dataset, X-Ray, About.
 * 2. Medical Report Analyzer: PDF/Image drag-and-drop, sample loading, in-memory extraction,
 *    lab test parameter parsing, high/low range checks, and sonography decoding.
 * 3. Original Abstract Classifier: TF-IDF feature extraction, Naive Bayes/Logistic Regression inference,
 *    and live probability bars.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ------------------------------------------------------------------------
    // Industry-Ready Global Toast Notification Utility
    // ------------------------------------------------------------------------
    function showToast(message, type = 'info', duration = 3500) {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        let icon = 'ℹ️';
        if (type === 'success') icon = '✅';
        else if (type === 'error') icon = '❌';
        else if (type === 'warning') icon = '⚠️';

        toast.innerHTML = `<span style="font-size: 1.1rem; flex-shrink: 0;">${icon}</span><span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(15px) scale(0.95)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // ------------------------------------------------------------------------
    // 1. Unified Navigation / Tab Switching (Desktop & Mobile Synchronized)
    // ------------------------------------------------------------------------
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const mobileHeaderActiveTab = document.getElementById('mobile-header-active-tab');

    const TAB_TITLES = {
        'tab-analyzer': '🩺 Report Analyzer',
        'tab-classify': '🧠 NLP Classifier',
        'tab-performance': '📊 Performance',
        'tab-dataset': '📚 Medical Corpus',
        'tab-xray': '🗺️ X-Ray Vision',
        'tab-about': 'ℹ️ About Project'
    };

    function switchActiveTab(targetTabId) {
        if (!targetTabId) return;

        // Synchronize all matching nav buttons (both desktop and mobile bottom bar)
        navButtons.forEach(b => {
            if (b.getAttribute('data-tab') === targetTabId) {
                b.classList.add('active');
            } else {
                b.classList.remove('active');
            }
        });

        // Toggle Tab Panes
        tabPanes.forEach(p => {
            if (p.id === targetTabId) {
                p.classList.add('active');
            } else {
                p.classList.remove('active');
            }
        });

        // Update Mobile Header Active Label
        if (mobileHeaderActiveTab && TAB_TITLES[targetTabId]) {
            mobileHeaderActiveTab.textContent = TAB_TITLES[targetTabId];
        }

        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    navButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const targetTabId = btn.getAttribute('data-tab');
            switchActiveTab(targetTabId);
        });
    });

    // ========================================================================
    // 2. MEDICAL REPORT ANALYZER (PHASE 3 & PHASE 6)
    // ========================================================================
    const reportDropzone = document.getElementById('report-dropzone');
    const reportFileInput = document.getElementById('report-file-input');
    const fileLoadedBadge = document.getElementById('file-loaded-badge');
    const loadedFilename = document.getElementById('loaded-filename');
    const loadedFilesize = document.getElementById('loaded-filesize');
    const removeFileBtn = document.getElementById('remove-file-btn');

    const reportTypeSelector = document.getElementById('report-type-selector');
    // Report Action Buttons (Print PDF & Copy Summary)
    const reportActionButtons = document.getElementById('report-action-buttons');
    const printReportBtn = document.getElementById('print-report-btn');
    const copySummaryBtn = document.getElementById('copy-summary-btn');

    if (printReportBtn) {
        printReportBtn.addEventListener('click', () => {
            window.print();
        });
    }

    // Helper function to build formatted clinical report for WhatsApp & Clipboard
    function buildFormattedClinicalReport(isWhatsApp = false) {
        const dateStr = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
        const activeTab = reportTypeSelector.value;
        const typeName = activeTab === 'lab' ? 'Laboratory Pathology Panel' : 'Abdominal Sonography Report';

        let lines = [];
        if (isWhatsApp) {
            lines.push(`*🏥 MEDREPORT AI — CLINICAL PATIENT SUMMARY*`);
            lines.push(`📅 *Date:* ${dateStr}`);
            lines.push(`📋 *Investigation:* ${typeName}\n`);
        } else {
            lines.push(`=== MEDREPORT AI - CLINICAL SUMMARY ===`);
            lines.push(`Date: ${dateStr}`);
            lines.push(`Report Type: ${typeName}`);
        }

        const normalBox = document.getElementById('normal-reassurance-box');
        const directBox = document.getElementById('direct-conditions-box');
        const sonoNormalBox = document.getElementById('sono-normal-reassurance-box');
        const sonoDirectBox = document.getElementById('sono-direct-conditions-box');

        const isTrulyNormal = (normalBox && normalBox.classList.contains('active-visible')) ||
                              (sonoNormalBox && sonoNormalBox.classList.contains('active-visible'));

        if (isTrulyNormal) {
            if (isWhatsApp) {
                lines.push(`🟢 *STATUS: 100% HEALTHY — NO DOCTOR NEEDED*`);
                lines.push(`✅ All tested diagnostic parameters fall well within standard physiological healthy intervals.`);
                lines.push(`💡 *Advice:* Maintain your regular healthy diet, exercise, and active hydration.`);
            } else {
                lines.push(`Status: 100% HEALTHY - NO DOCTOR NEEDED`);
                lines.push(`All tested parameters fall within standard healthy reference intervals.`);
            }
        } else {
            const activeCondBox = activeTab === 'lab' ? directBox : sonoDirectBox;
            const condItems = activeCondBox ? activeCondBox.querySelectorAll('.direct-cond-item') : [];

            if (isWhatsApp) {
                lines.push(`🔴 *STATUS: CLINICAL ATTENTION NEEDED*`);
                if (condItems.length > 0) {
                    lines.push(`\n⚠️ *Identified Diseases & Pathologies (${condItems.length}):*`);
                    condItems.forEach(it => {
                        const title = it.querySelector('.direct-cond-title')?.textContent.trim();
                        const act = it.querySelector('.direct-cond-action')?.textContent.trim();
                        if (title) lines.push(`• *${title}*`);
                        if (act) lines.push(`   👉 _Action:_ ${act}`);
                    });
                }
            } else {
                lines.push(`Status: CLINICAL ATTENTION NEEDED`);
                if (condItems.length > 0) {
                    lines.push(`Detected Conditions (${condItems.length}):`);
                    condItems.forEach(it => {
                        const title = it.querySelector('.direct-cond-title')?.textContent.trim();
                        const act = it.querySelector('.direct-cond-action')?.textContent.trim();
                        if (title) lines.push(`- ${title}`);
                        if (act) lines.push(`  Action: ${act}`);
                    });
                }
            }

            // Key impressions
            if (activeTab === 'lab') {
                const impression = document.getElementById('diagnosis-impression-text')?.textContent.trim();
                if (impression) {
                    lines.push(isWhatsApp ? `\n🩺 *Diagnostic Impression:*\n${impression}` : `Clinical Impression: ${impression}`);
                }
            } else {
                const sonoImp = document.getElementById('sono-impression-text')?.textContent.trim();
                if (sonoImp) {
                    lines.push(isWhatsApp ? `\n📡 *Radiological Impression:*\n${sonoImp}` : `Radiological Impression: ${sonoImp}`);
                }
            }
        }

        if (isWhatsApp) {
            lines.push(`\n_⚠️ Note: Automated AI summary for physician verification. Not an unconditional prescription._`);
            lines.push(`🌐 _Verified via MedReport AI Platform_`);
        } else {
            lines.push(`\nNotice: For clinical verification only. Consult a certified medical doctor.`);
        }

        return lines.join('\n');
    }

    // WhatsApp Share Button Handler
    const whatsappShareBtn = document.getElementById('whatsapp-share-btn');
    if (whatsappShareBtn) {
        whatsappShareBtn.addEventListener('click', () => {
            try {
                const reportMsg = buildFormattedClinicalReport(true);
                const waUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent(reportMsg)}`;
                window.open(waUrl, '_blank');
                showToast('Opening WhatsApp with clinical summary...', 'success', 3000);
            } catch (err) {
                showToast('Could not open WhatsApp', 'error');
            }
        });
    }

    // Abstract Tab WhatsApp Share Button Handler
    const abstractWhatsappBtn = document.getElementById('abstract-whatsapp-btn');
    if (abstractWhatsappBtn) {
        abstractWhatsappBtn.addEventListener('click', () => {
            try {
                const catName = document.getElementById('res-category-name')?.textContent || '';
                const catDesc = document.getElementById('res-category-desc')?.textContent || '';
                const model = document.getElementById('res-model-name')?.textContent || '';

                const lines = [
                    `*🧠 MEDREPORT AI — NLP CLASSIFICATION REPORT*`,
                    `📅 *Date:* ${new Date().toLocaleDateString('en-GB')}`,
                    `\n🎯 *Predicted Category:* ${catName}`,
                    `📖 *Clinical Scope:* ${catDesc}`,
                    `⚙️ *Model Engine:* ${model}`,
                    `\n_⚠️ Generated via MedReport AI Biomedical NLP Classifier_`
                ];

                const waUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent(lines.join('\n'))}`;
                window.open(waUrl, '_blank');
                showToast('Opening WhatsApp with classification...', 'success', 3000);
            } catch (err) {
                showToast('Could not open WhatsApp', 'error');
            }
        });
    }


    if (copySummaryBtn) {
        copySummaryBtn.addEventListener('click', async () => {
            try {
                const textToCopy = buildFormattedClinicalReport(false);
                await navigator.clipboard.writeText(textToCopy);
                showToast('Diagnostic summary copied to clipboard!', 'success');
            } catch (err) {
                showToast('Failed to copy summary to clipboard', 'error');
            }
        });
    }

    const reportLangSelector = document.getElementById('report-lang-selector');
    const reportSampleButtons = document.getElementById('report-sample-buttons');
    const reportTextInput = document.getElementById('report-text-input');
    const reportCharCounter = document.getElementById('report-char-counter');
    const reportWordCounter = document.getElementById('report-word-counter');

    const analyzeReportBtn = document.getElementById('analyze-report-btn');
    const clearReportBtn = document.getElementById('clear-report-btn');
    const reportBtnText = analyzeReportBtn.querySelector('.btn-text');
    const reportBtnSpinner = analyzeReportBtn.querySelector('.spinner');

    const reportError = document.getElementById('report-error');
    const reportErrorMsg = document.getElementById('report-error-msg');

    const reportResultsEmpty = document.getElementById('report-results-empty');
    const labResultsContent = document.getElementById('lab-results-content');
    const sonographyResultsContent = document.getElementById('sonography-results-content');
    const reportLiveBadge = document.getElementById('report-live-badge');
    const reportLangPill = document.getElementById('report-lang-pill');

    // Lab Results elements
    const labTotalCount = document.getElementById('lab-total-count');
    const labHighCount = document.getElementById('lab-high-count');
    const labLowCount = document.getElementById('lab-low-count');
    const labNormalCount = document.getElementById('lab-normal-count');
    const labTableBody = document.getElementById('lab-table-body');
    const clinicalDiagnosisBox = document.getElementById('clinical-diagnosis-box');
    const clinicalAcuityBadge = document.getElementById('clinical-acuity-badge');
    const diagImpressionsList = document.getElementById('diag-impressions-list');
    const diagDifferentialsList = document.getElementById('diag-differentials-list');
    const diagNextStepsList = document.getElementById('diag-next-steps-list');

    // Sonography Results elements
    const sonoImpressionText = document.getElementById('sono-impression-text');
    const sonoObsList = document.getElementById('sono-obs-list');
    const sonoTermsGrid = document.getElementById('sono-terms-grid');

    // Update Report Text Counters
    function updateReportCounters() {
        const text = reportTextInput.value;
        if (reportCharCounter) reportCharCounter.textContent = `${text.length.toLocaleString()} characters`;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        if (reportWordCounter) reportWordCounter.textContent = `${words.toLocaleString()} words`;
        const compactBadge = document.getElementById('report-char-count');
        if (compactBadge) compactBadge.textContent = `${words.toLocaleString()} words | ${text.length.toLocaleString()} chars`;
        if (text.trim().length > 0) {
            hideReportError();
        }
    }
    reportTextInput.addEventListener('input', updateReportCounters);

    function showReportError(msg) {
        reportErrorMsg.textContent = msg;
        reportError.style.display = 'flex';
        reportTextInput.focus();
    }

    function hideReportError() {
        reportError.style.display = 'none';
    }

    const pasteReportBtn = document.getElementById('paste-report-btn');
    const reportSampleFilters = document.getElementById('report-sample-filters');
    let allReportSamples = [];
    let activeReportFilter = 'all';

    // Clipboard Paste Helper for Report Text
    if (pasteReportBtn) {
        pasteReportBtn.addEventListener('click', async () => {
            try {
                if (!navigator.clipboard || !navigator.clipboard.readText) {
                    reportTextInput.focus();
                    return;
                }
                const text = await navigator.clipboard.readText();
                if (text && text.trim().length > 0) {
                    reportTextInput.value = text;
                    updateReportCounters();
                    hideReportError();
                    const origHtml = pasteReportBtn.innerHTML;
                    pasteReportBtn.innerHTML = `
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                        <span>Pasted!</span>
                    `;
                    pasteReportBtn.classList.add('btn-pasted');
                    setTimeout(() => {
                        pasteReportBtn.innerHTML = origHtml;
                        pasteReportBtn.classList.remove('btn-pasted');
                    }, 1800);
                }
            } catch (err) {
                reportTextInput.focus();
            }
        });
    }

    // Render Filtered Report Sample Buttons
    function renderReportSampleButtons(filter = 'all') {
        activeReportFilter = filter.toLowerCase();
        reportSampleButtons.innerHTML = '';

        const filtered = allReportSamples.filter(s => {
            if (activeReportFilter === 'all') return true;
            const lang = (s.language || '').toLowerCase();
            const title = (s.title || '').toLowerCase();
            return lang === activeReportFilter || title.includes(`[${activeReportFilter}]`);
        });

        if (filtered.length === 0) {
            reportSampleButtons.innerHTML = '<span class="sample-empty-note">No samples available for this filter</span>';
            return;
        }

        filtered.forEach(s => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'sample-pill';

            let icon = '🔬';
            if (s.type === 'sonography') icon = '📡';
            else if ((s.id || '').includes('lipid') || (s.title || '').toLowerCase().includes('lipid')) icon = '❤️';
            else if ((s.id || '').includes('cmp') || (s.title || '').toLowerCase().includes('metabolic')) icon = '🧪';

            // Clean title removing [Language] prefix
            const cleanTitle = s.title.replace(/^\[.*?\]\s*/, '');
            const langBadge = s.language && s.language !== 'English' && activeReportFilter === 'all' ? `[${s.language}] ` : '';

            btn.innerHTML = `<span>${icon}</span> <span>${langBadge}${cleanTitle}</span>`;
            btn.title = s.title;

            btn.addEventListener('click', () => {
                reportTypeSelector.value = s.type;
                reportTextInput.value = s.snippet;
                updateReportCounters();
                hideReportError();
                resetReportBadge();
                reportTextInput.focus();
            });

            reportSampleButtons.appendChild(btn);
        });
    }

    if (reportSampleFilters) {
        reportSampleFilters.addEventListener('click', (e) => {
            const btn = e.target.closest('.sample-filter-btn');
            if (!btn) return;
            reportSampleFilters.querySelectorAll('.sample-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderReportSampleButtons(btn.dataset.filter || 'all');
        });
    }

    // Load Sample Reports from Backend
    async function loadSampleReports() {
        try {
            const res = await fetch('/api/report/sample-reports');
            if (!res.ok) throw new Error();
                        allReportSamples = await res.json();
            populateReportSampleDropdown();
            renderReportSampleButtons(activeReportFilter);
        } catch (e) {
            reportSampleButtons.innerHTML = '<span class="sample-pill-loading">Failed to load report samples</span>';
        }
    }
    
    // Populate Quick Test Sample Dropdown (Space-Saving Arrow Down Selector)
    function populateReportSampleDropdown() {
        const select = document.getElementById('report-sample-select');
        if (!select) return;
        select.innerHTML = '<option value="">Select a sample report... ▼</option>';
        allReportSamples.forEach((s, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            let icon = s.type === 'sonography' ? '📡' : '🔬';
            opt.textContent = `${icon} ${s.title}`;
            select.appendChild(opt);
        });

        select.addEventListener('change', () => {
            const val = select.value;
            if (val !== '' && allReportSamples[val]) {
                const s = allReportSamples[val];
                reportTextInput.value = s.text;
                updateReportCounters();
                hideReportError();
                if (s.type) reportTypeSelector.value = s.type;
                if (s.language && reportLangSelector) {
                    const l = s.language.toLowerCase();
                    if (['english', 'hinglish', 'gujlish'].includes(l)) {
                        reportLangSelector.value = l;
                    }
                }
                showToast(`Loaded: ${s.title}`, 'info', 2200);
            }
        });
    }

    // Toggle Collapsible Sample Pills Tray with Arrow Rotation
    const reportSampleToggle = document.getElementById('report-sample-toggle');
    if (reportSampleToggle) {
        reportSampleToggle.addEventListener('click', () => {
            const pills = document.getElementById('report-sample-buttons');
            if (pills) {
                const isCollapsed = pills.classList.toggle('collapsed');
                reportSampleToggle.classList.toggle('expanded', !isCollapsed);
                const txt = reportSampleToggle.querySelector('.toggle-text');
                if (txt) txt.textContent = isCollapsed ? 'All Samples' : 'Hide Samples';
            }
        });
    }

    loadSampleReports();

    // Dropzone Click & Drag Events
    reportDropzone.addEventListener('click', () => reportFileInput.click());

    ['dragenter', 'dragover'].forEach(evt => {
        reportDropzone.addEventListener(evt, e => {
            e.preventDefault();
            reportDropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        reportDropzone.addEventListener(evt, e => {
            e.preventDefault();
            reportDropzone.classList.remove('dragover');
        });
    });

    reportDropzone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    reportFileInput.addEventListener('change', () => {
        if (reportFileInput.files.length > 0) {
            handleFileUpload(reportFileInput.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetReportBadge();
        reportTextInput.value = '';
        updateReportCounters();
    });

    function resetReportBadge() {
        fileLoadedBadge.style.display = 'none';
        reportDropzone.style.display = 'block';
        reportFileInput.value = '';
    }

    // Handle File Upload to /api/report/upload
    async function handleFileUpload(file) {
        // Validation
        const validExtensions = ['.pdf', '.png', '.jpg', '.jpeg'];
        const fileName = file.name.toLowerCase();
        const isValid = validExtensions.some(ext => fileName.endsWith(ext));

        if (!isValid) {
            showReportError('Unsupported file type. Please upload a PDF, PNG, or JPG report.');
            return;
        }

        if (file.size > 15 * 1024 * 1024) {
            showReportError('File exceeds 15 MB limit. Please upload a smaller report.');
            return;
        }

        hideReportError();

        // Show File Badge
        loadedFilename.textContent = file.name;
        loadedFilesize.textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
        fileLoadedBadge.style.display = 'flex';
        reportDropzone.style.display = 'none';

        reportBtnText.textContent = 'Extracting Text...';
        reportBtnSpinner.style.display = 'inline-block';
        analyzeReportBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/api/report/upload', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Failed to extract text from file.');

            reportTextInput.value = data.extracted_text;
            updateReportCounters();
            showToast(`Report extracted successfully from ${file.name}`, 'success');

            // Auto-detect type if sonography/ultrasound mentioned
            if (data.extracted_text.toLowerCase().includes('ultrasound') || data.extracted_text.toLowerCase().includes('sonography')) {
                reportTypeSelector.value = 'sonography';
            } else {
                reportTypeSelector.value = 'lab';
            }

        } catch (err) {
            showReportError(err.message || 'Error occurred while processing file.');
            showToast(err.message || 'Error processing report file', 'error');
            resetReportBadge();
        } finally {
            analyzeReportBtn.disabled = false;
            reportBtnText.textContent = 'Analyze Report';
            reportBtnSpinner.style.display = 'none';
        }
    }

    // Clear Report
    clearReportBtn.addEventListener('click', () => {
        reportTextInput.value = '';
        resetReportBadge();
        updateReportCounters();
        hideReportError();

        reportResultsEmpty.style.display = 'block';
        if (reportActionButtons) reportActionButtons.style.display = 'none';
        labResultsContent.style.display = 'none';
        sonographyResultsContent.style.display = 'none';
        if (clinicalDiagnosisBox) clinicalDiagnosisBox.style.display = 'none';
        if (reportLangPill) reportLangPill.style.display = 'none';

        const normalBox = document.getElementById('normal-reassurance-box');
        if (normalBox) {
            normalBox.classList.remove('active-visible');
            normalBox.style.setProperty('display', 'none', 'important');
        }
        const directBox = document.getElementById('direct-conditions-box');
        if (directBox) {
            directBox.classList.remove('active-visible');
            directBox.style.setProperty('display', 'none', 'important');
        }
        const sonoNormalBox = document.getElementById('sono-normal-reassurance-box');
        if (sonoNormalBox) {
            sonoNormalBox.classList.remove('active-visible');
            sonoNormalBox.style.setProperty('display', 'none', 'important');
        }
        const sonoDirectBox = document.getElementById('sono-direct-conditions-box');
        if (sonoDirectBox) {
            sonoDirectBox.classList.remove('active-visible');
            sonoDirectBox.style.setProperty('display', 'none', 'important');
        }

        reportLiveBadge.textContent = 'READY';
        reportLiveBadge.classList.remove('live');
        reportTextInput.focus();
    });

    // Analyze Report Button
    analyzeReportBtn.addEventListener('click', async () => {
        const text = reportTextInput.value.trim();
        if (!text) {
            showReportError('Please provide medical report text to analyze.');
            return;
        }

        hideReportError();

        analyzeReportBtn.disabled = true;
        reportBtnText.textContent = 'Analyzing Report...';
        reportBtnSpinner.style.display = 'inline-block';
        reportLiveBadge.textContent = 'PROCESSING';
        reportLiveBadge.classList.add('live');

        const mode = reportTypeSelector.value;
        const endpoint = mode === 'lab' ? '/api/report/analyze-lab' : '/api/report/analyze-sonography';
        const targetLang = reportLangSelector ? reportLangSelector.value : 'auto';

        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ report_text: text, language: targetLang })
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Analysis failed on the server.');

            reportResultsEmpty.style.display = 'none';
            reportLiveBadge.textContent = 'ANALYZED';
            if (reportActionButtons) reportActionButtons.style.display = 'inline-flex';
            showToast('Clinical report analyzed successfully!', 'success');

            if (reportLangPill && data.detected_language) {
                reportLangPill.style.display = 'inline-flex';
                reportLangPill.textContent = `🌐 ${data.detected_language.language_name.split('(')[0].trim()}`;
            }

            if (mode === 'lab') {
                renderLabResults(data);
            } else {
                renderSonographyResults(data);
            }

            // On mobile / tablet screens, smoothly auto-scroll down to results
            if (window.innerWidth < 1024) {
                setTimeout(() => {
                    const resPanel = document.getElementById('report-results-panel');
                    if (resPanel) {
                        resPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 100);
            }

        } catch (err) {
            showReportError(err.message || 'Network error occurred during analysis.');
            reportLiveBadge.textContent = 'ERROR';
            reportLiveBadge.classList.remove('live');
        } finally {
            analyzeReportBtn.disabled = false;
            reportBtnText.textContent = 'Analyze Report';
            reportBtnSpinner.style.display = 'none';
        }
    });

    // Helper to render visual reference range gauge
    function renderRangeGauge(val, low, high, status) {
        const numVal = parseFloat(val);
        const numLow = parseFloat(low);
        const numHigh = parseFloat(high);

        if (isNaN(numVal) || isNaN(numLow) || isNaN(numHigh) || numHigh <= numLow) {
            return '';
        }

        const span = numHigh - numLow;
        const minDisp = Math.max(0, numLow - span * 0.35);
        const maxDisp = numHigh + span * 0.35;
        const totalDisp = maxDisp - minDisp;
        if (totalDisp <= 0) return '';

        const normStartPct = Math.max(0, Math.min(100, ((numLow - minDisp) / totalDisp) * 100));
        const normWidthPct = Math.max(0, Math.min(100 - normStartPct, ((numHigh - numLow) / totalDisp) * 100));
        const markerPct = Math.max(2, Math.min(98, ((numVal - minDisp) / totalDisp) * 100));

        const statusClass = (status || 'normal').toLowerCase();

        return `
            <div class="mini-gauge" title="Normal Range: ${numLow} - ${numHigh} | Measured: ${numVal}">
                <div class="gauge-track">
                    <div class="gauge-zone-normal" style="left: ${normStartPct.toFixed(1)}%; width: ${normWidthPct.toFixed(1)}%;"></div>
                    <div class="gauge-marker marker-${statusClass}" style="left: ${markerPct.toFixed(1)}%;">
                        <div class="marker-dot"></div>
                    </div>
                </div>
                <div class="gauge-bounds">
                    <span class="bound-lbl">${numLow}</span>
                    <span class="bound-lbl bound-center">Normal</span>
                    <span class="bound-lbl">${numHigh}</span>
                </div>
            </div>
        `;
    }

    // Helper to render Directly Identified Conditions (Dengue, Malaria, Typhoid, Diabetes, etc.)
    function renderDirectConditions(conditions, activeLang, boxId, countId, listId) {
        const box = document.getElementById(boxId);
        if (!box) return;

        if (!conditions || conditions.length === 0) {
            box.classList.remove('active-visible');
            box.style.setProperty('display', 'none', 'important');
            return;
        }

        box.classList.add('active-visible');
        box.style.setProperty('display', 'block', 'important');
        const countBadge = document.getElementById(countId);
        if (countBadge) {
            countBadge.textContent = `${conditions.length} Condition${conditions.length > 1 ? 's' : ''} Identified`;
        }

        const list = document.getElementById(listId);
        if (!list) return;
        list.innerHTML = '';

        conditions.forEach(c => {
            const item = document.createElement('div');
            const sevClass = (c.severity || 'high').toLowerCase();
            item.className = `direct-condition-item severity-${sevClass}`;

            const summary = (c.summary && (c.summary[activeLang] || c.summary['en'])) || '';
            const rec = (c.recommendation && (c.recommendation[activeLang] || c.recommendation['en'])) || '';

            item.innerHTML = `
                <div class="direct-cond-top">
                    <div class="cond-name-wrap">
                        <span class="cond-icon">${c.icon || '⚠️'}</span>
                        <strong class="cond-heading">${c.name}</strong>
                    </div>
                    <span class="cond-status-tag tag-${sevClass}">${c.badge}</span>
                </div>
                <p class="cond-desc">${summary}</p>
                <div class="cond-action-row">
                    <span class="action-tag">👉 Clinical Action:</span>
                    <span class="action-text">${rec}</span>
                </div>
            `;
            list.appendChild(item);
        });
    }

    // Render Laboratory Results
    function renderLabResults(data) {
        sonographyResultsContent.style.display = 'none';
        labResultsContent.style.display = 'block';

        labTotalCount.textContent = data.total_parameters_detected || 0;
        labHighCount.textContent = data.high_count || 0;
        labLowCount.textContent = data.low_count || 0;
        labNormalCount.textContent = data.normal_count || 0;

        // 1. Strict Mutual Exclusivity: Reassurance Box vs Directly Identified Conditions
        const normalBox = document.getElementById('normal-reassurance-box');
        const directBox = document.getElementById('direct-conditions-box');

        const hasDirectConditions = Array.isArray(data.directly_identified_conditions) && data.directly_identified_conditions.length > 0;
        const hasAbnormalTests = (data.high_count > 0) || (data.low_count > 0);
        const isTrulyNormal = Boolean(data.is_all_normal && data.normal_reassurance && !hasDirectConditions && !hasAbnormalTests);

        if (normalBox) {
            if (isTrulyNormal) {
                normalBox.classList.add('active-visible');
                normalBox.style.setProperty('display', 'flex', 'important');
                const nTitle = document.getElementById('normal-reassurance-title');
                const nMsg = document.getElementById('normal-reassurance-msg');
                const nRec = document.getElementById('normal-reassurance-rec');
                if (nTitle) nTitle.textContent = data.normal_reassurance.title;
                if (nMsg) nMsg.textContent = data.normal_reassurance.message;
                if (nRec) nRec.textContent = data.normal_reassurance.recommendation;
            } else {
                normalBox.classList.remove('active-visible');
                normalBox.style.setProperty('display', 'none', 'important');
            }
        }

        // 2. Directly Identified Conditions Box (Dengue, Malaria, Typhoid, Diabetes, Jaundice, etc.)
        if (isTrulyNormal) {
            if (directBox) {
                directBox.classList.remove('active-visible');
                directBox.style.setProperty('display', 'none', 'important');
            }
        } else {
            renderDirectConditions(data.directly_identified_conditions, data.active_language || 'en', 'direct-conditions-box', 'direct-cond-counter', 'direct-conditions-list');
        }

        labTableBody.innerHTML = '';

        if (!data.tests || data.tests.length === 0) {
            labTableBody.innerHTML = `
                <tr>
                    <td colspan="5" style="text-align: center; padding: 24px; color: var(--text-muted);">
                        No structured laboratory parameters were matched. Please verify that your report text contains test names with numerical values (e.g. "Hemoglobin: 10.5 g/dL").
                    </td>
                </tr>
            `;
            return;
        }

        data.tests.forEach(t => {
            const tr = document.createElement('tr');
            tr.className = `lab-row-${(t.status || 'normal').toLowerCase()}`;
            const gaugeHtml = renderRangeGauge(t.measured_value, t.ref_low, t.ref_high, t.status);

            tr.innerHTML = `
                <td>
                    <div class="test-name-cell">
                        <strong>${t.test_name}</strong>
                        <span class="range-source-tag">${t.category}</span>
                    </div>
                </td>
                <td>
                    <span class="val-highlight val-${(t.status || 'normal').toLowerCase()}">${t.measured_value}</span>
                    <span class="val-unit">${t.unit}</span>
                </td>
                <td>
                    <div class="ref-range-cell">
                        <div class="ref-text">${t.ref_range_str}</div>
                        ${gaugeHtml}
                        <span class="range-source-tag">(${t.range_source})</span>
                    </div>
                </td>
                <td>
                    <span class="badge-status ${t.badge_class}">${t.status}</span>
                </td>
                <td>
                    <p class="lab-expl">${t.plain_explanation}</p>
                </td>
            `;
            labTableBody.appendChild(tr);
        });

        // Render Clinical Diagnostic Synthesis
        if (clinicalDiagnosisBox && data.clinical_diagnosis) {
            clinicalDiagnosisBox.style.display = 'block';
            const diag = data.clinical_diagnosis;

            // 1. Acuity Badge
            const acuity = (diag.acuity_level || 'NORMAL').toUpperCase();
            clinicalAcuityBadge.textContent = acuity;
            clinicalAcuityBadge.className = `acuity-badge acuity-${acuity.toLowerCase()}`;

            // 2. Primary Impressions
            diagImpressionsList.innerHTML = '';
            if (diag.primary_impressions && diag.primary_impressions.length > 0) {
                diag.primary_impressions.forEach(imp => {
                    const li = document.createElement('li');
                    li.className = 'diag-impression-item';
                    li.innerHTML = `<span class="bullet-dot"></span><span>${imp}</span>`;
                    diagImpressionsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.className = 'diag-impression-item normal-state';
                li.innerHTML = '<span class="bullet-dot green"></span><span>All mapped parameters fall within standard reference intervals. No active syndromic acute abnormalities detected.</span>';
                diagImpressionsList.appendChild(li);
            }

            // 3. Differential Diagnoses
            diagDifferentialsList.innerHTML = '';
            if (diag.differential_diagnoses && diag.differential_diagnoses.length > 0) {
                diag.differential_diagnoses.forEach(diff => {
                    const chip = document.createElement('span');
                    chip.className = 'differential-chip';
                    chip.textContent = diff;
                    diagDifferentialsList.appendChild(chip);
                });
            } else {
                diagDifferentialsList.innerHTML = '<span class="empty-diff-note">No differential diagnoses indicated based on current values.</span>';
            }

            // 4. Recommended Next Steps
            diagNextStepsList.innerHTML = '';
            if (diag.recommended_next_steps && diag.recommended_next_steps.length > 0) {
                diag.recommended_next_steps.forEach(step => {
                    const li = document.createElement('li');
                    li.className = 'diag-step-item';
                    li.innerHTML = `<span class="step-arrow">→</span><span>${step}</span>`;
                    diagNextStepsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.className = 'diag-step-item normal-state';
                li.innerHTML = '<span class="step-arrow">✓</span><span>Maintain regular health check-ups and standard preventive health measures.</span>';
                diagNextStepsList.appendChild(li);
            }
        }
    }

    // Render Sonography Results
    function renderSonographyResults(data) {
        labResultsContent.style.display = 'none';
        sonographyResultsContent.style.display = 'block';

        // 1. Strict Mutual Exclusivity for Sonography (Normal Reassurance vs Identified Conditions)
        const sonoNormalBox = document.getElementById('sono-normal-reassurance-box');
        const sonoDirectBox = document.getElementById('sono-direct-conditions-box');

        const hasSonoDirect = Array.isArray(data.directly_identified_conditions) && data.directly_identified_conditions.length > 0;
        const hasSonoAttention = Array.isArray(data.clinical_observations) && data.clinical_observations.some(o => o.severity === 'ATTENTION');
        const isTrulySonoNormal = Boolean(data.is_all_normal && data.normal_reassurance && !hasSonoDirect && !hasSonoAttention);

        if (sonoNormalBox) {
            if (isTrulySonoNormal) {
                sonoNormalBox.classList.add('active-visible');
                sonoNormalBox.style.setProperty('display', 'flex', 'important');
                const sTitle = document.getElementById('sono-normal-title');
                const sMsg = document.getElementById('sono-normal-msg');
                const sRec = document.getElementById('sono-normal-rec');
                if (sTitle) sTitle.textContent = data.normal_reassurance.title;
                if (sMsg) sMsg.textContent = data.normal_reassurance.message;
                if (sRec) sRec.textContent = data.normal_reassurance.recommendation;
            } else {
                sonoNormalBox.classList.remove('active-visible');
                sonoNormalBox.style.setProperty('display', 'none', 'important');
            }
        }

        // 2. Directly Identified Conditions for Sonography (Gallstones, Fatty Liver, Kidney Stones)
        if (isTrulySonoNormal) {
            if (sonoDirectBox) {
                sonoDirectBox.classList.remove('active-visible');
                sonoDirectBox.style.setProperty('display', 'none', 'important');
            }
        } else {
            renderDirectConditions(data.directly_identified_conditions, data.active_language || 'en', 'sono-direct-conditions-box', 'sono-direct-cond-counter', 'sono-direct-conditions-list');
        }

        sonoImpressionText.textContent = data.radiological_impression || 'No impression found.';

        // Render Organ Observations
        sonoObsList.innerHTML = '';
        if (data.clinical_observations && data.clinical_observations.length > 0) {
            data.clinical_observations.forEach(obs => {
                const card = document.createElement('div');
                const sevClass = obs.severity === 'ATTENTION' ? 'attention' : (obs.severity === 'NORMAL' ? 'normal' : 'info');
                card.className = `sono-obs-card ${sevClass}`;

                card.innerHTML = `
                    <div class="sono-obs-header">
                        <span class="sono-organ-name">${obs.organ}</span>
                        <span class="badge-status ${obs.severity === 'ATTENTION' ? 'badge-high' : 'badge-normal'}">${obs.severity}</span>
                    </div>
                    <div class="sono-finding-title">${obs.finding}</div>
                    <p class="sono-plain-desc">${obs.plain_meaning}</p>
                `;
                sonoObsList.appendChild(card);
            });
        } else {
            sonoObsList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.86rem;">No specific organ descriptors matched in the text.</p>';
        }

        // Render Decoded Terminology
        sonoTermsGrid.innerHTML = '';
        if (data.detected_terms && data.detected_terms.length > 0) {
            data.detected_terms.forEach(term => {
                const card = document.createElement('div');
                card.className = 'sono-term-card';
                card.innerHTML = `
                    <div class="sono-term-title">${term.term}</div>
                    <p class="sono-term-desc">${term.meaning}</p>
                `;
                sonoTermsGrid.appendChild(card);
            });
        } else {
            sonoTermsGrid.innerHTML = '<p style="color: var(--text-muted); font-size: 0.86rem; grid-column: span 2;">No specialized ultrasound acoustics terminology (hyperechoic/anechoic) identified in the text.</p>';
        }
    }


    // ========================================================================
    // 3. ORIGINAL MEDICAL ABSTRACT CLASSIFIER (PHASE 4 - PRESERVED 100%)
    // ========================================================================
    const abstractInput = document.getElementById('abstract-input');
    const charCounter = document.getElementById('char-counter');
    const wordCounter = document.getElementById('word-counter');
    const modelSelector = document.getElementById('model-selector');
    const sampleButtonsContainer = document.getElementById('sample-buttons');

    const classifyBtn = document.getElementById('classify-btn');
    const clearBtn = document.getElementById('clear-btn');
    const btnText = classifyBtn.querySelector('.btn-text');
    const btnSpinner = classifyBtn.querySelector('.spinner');

    const inputError = document.getElementById('input-error');
    const inputErrorMsg = document.getElementById('input-error-msg');

    const resultsEmpty = document.getElementById('results-empty');
    const resultsContent = document.getElementById('results-content');
    const liveBadge = document.getElementById('live-badge');
    const classifyLangPill = document.getElementById('classify-lang-pill');

    const resCategoryName = document.getElementById('res-category-name');
    const resCategoryDesc = document.getElementById('res-category-desc');
    const resCategoryId = document.getElementById('res-category-id');
    const resModelName = document.getElementById('res-model-name');
    const resTokenCount = document.getElementById('res-token-count');
    const resMultilingualBanner = document.getElementById('res-multilingual-banner');
    const resDetectedLangTag = document.getElementById('res-detected-lang-tag');
    const resTransStatus = document.getElementById('res-trans-status');
    const resTransQuery = document.getElementById('res-trans-query');
    const probBarsList = document.getElementById('prob-bars-list');

    const pasteAbstractBtn = document.getElementById('paste-abstract-btn');
    const abstractSampleFilters = document.getElementById('abstract-sample-filters');
    let allAbstractSamples = [];
    let activeAbstractFilter = 'all';

    // Clipboard Paste Helper for Abstract
    if (pasteAbstractBtn) {
        pasteAbstractBtn.addEventListener('click', async () => {
            try {
                if (!navigator.clipboard || !navigator.clipboard.readText) {
                    abstractInput.focus();
                    return;
                }
                const text = await navigator.clipboard.readText();
                if (text && text.trim().length > 0) {
                    abstractInput.value = text;
                    updateAbstractCounters();
                    hideAbstractError();
                    const origHtml = pasteAbstractBtn.innerHTML;
                    pasteAbstractBtn.innerHTML = `
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                        <span>Pasted!</span>
                    `;
                    pasteAbstractBtn.classList.add('btn-pasted');
                    setTimeout(() => {
                        pasteAbstractBtn.innerHTML = origHtml;
                        pasteAbstractBtn.classList.remove('btn-pasted');
                    }, 1800);
                }
            } catch (err) {
                abstractInput.focus();
            }
        });
    }

    function updateAbstractCounters() {
        const text = abstractInput.value;
        charCounter.textContent = `${text.length.toLocaleString()} / 15,000 characters`;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        wordCounter.textContent = `${words.toLocaleString()} words`;
        if (text.trim().length > 0) hideAbstractError();
    }
    abstractInput.addEventListener('input', updateAbstractCounters);

    function showAbstractError(msg) {
        inputErrorMsg.textContent = msg;
        inputError.style.display = 'flex';
        abstractInput.focus();
    }

    function hideAbstractError() {
        inputError.style.display = 'none';
    }

    // Render Filtered Abstract Sample Buttons
    function renderAbstractSampleButtons(filter = 'all') {
        activeAbstractFilter = filter.toLowerCase();
        sampleButtonsContainer.innerHTML = '';

        const filtered = allAbstractSamples.filter(sample => {
            if (activeAbstractFilter === 'all') return true;
            const lang = (sample.language || '').toLowerCase();
            const title = (sample.title || '').toLowerCase();
            return lang === activeAbstractFilter || title.includes(`[${activeAbstractFilter}]`);
        });

        if (filtered.length === 0) {
            sampleButtonsContainer.innerHTML = '<span class="sample-empty-note">No samples available for this filter</span>';
            return;
        }

        filtered.forEach(sample => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'sample-pill';

            let icon = '📄';
            const cat = (sample.category || '').toLowerCase();
            if (cat.includes('neoplasm')) icon = '🎗️';
            else if (cat.includes('cardio')) icon = '❤️';
            else if (cat.includes('digestive')) icon = '🩺';

            const cleanTitle = sample.title.replace(/^\[.*?\]\s*/, '').trim();
            const langLabel = sample.language && sample.language !== 'English' && activeAbstractFilter === 'all' ? `[${sample.language}] ` : '';

            btn.innerHTML = `<span>${icon}</span> <span>${langLabel}${cleanTitle}</span>`;
            btn.title = `${sample.category} - ${sample.title}`;

            btn.addEventListener('click', () => {
                abstractInput.value = sample.snippet;
                updateAbstractCounters();
                hideAbstractError();
                abstractInput.focus();
            });

            sampleButtonsContainer.appendChild(btn);
        });
    }

    if (abstractSampleFilters) {
        abstractSampleFilters.addEventListener('click', (e) => {
            const btn = e.target.closest('.sample-filter-btn');
            if (!btn) return;
            abstractSampleFilters.querySelectorAll('.sample-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderAbstractSampleButtons(btn.dataset.filter || 'all');
        });
    }

    // Load Abstract Samples
    async function loadAbstractSamples() {
        try {
            const res = await fetch('/api/sample-texts');
            if (!res.ok) throw new Error();
                        allAbstractSamples = await res.json();
            populateAbstractSampleDropdown();
            renderAbstractSampleButtons(activeAbstractFilter);
        } catch (err) {
            sampleButtonsContainer.innerHTML = '<span class="sample-pill-loading">Failed to load samples</span>';
        }
    }
    
    // Populate Abstract Sample Dropdown (Space-Saving Arrow Down Selector)
    function populateAbstractSampleDropdown() {
        const select = document.getElementById('abstract-sample-select');
        if (!select) return;
        select.innerHTML = '<option value="">Select a sample abstract... ▼</option>';
        allAbstractSamples.forEach((s, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = `[${s.category}] ${s.title}`;
            select.appendChild(opt);
        });

        select.addEventListener('change', () => {
            const val = select.value;
            if (val !== '' && allAbstractSamples[val]) {
                const s = allAbstractSamples[val];
                abstractInput.value = s.text;
                updateAbstractCounters();
                hideAbstractError();
                showToast(`Loaded: ${s.title}`, 'info', 2200);
            }
        });
    }

    // Toggle Collapsible Abstract Sample Pills Tray with Arrow Rotation
    const abstractSampleToggle = document.getElementById('abstract-sample-toggle');
    if (abstractSampleToggle) {
        abstractSampleToggle.addEventListener('click', () => {
            const pills = document.getElementById('sample-buttons');
            if (pills) {
                const isCollapsed = pills.classList.toggle('collapsed');
                abstractSampleToggle.classList.toggle('expanded', !isCollapsed);
                const txt = abstractSampleToggle.querySelector('.toggle-text');
                if (txt) txt.textContent = isCollapsed ? 'All Samples' : 'Hide Samples';
            }
        });
    }

    loadAbstractSamples();

    // Clear Abstract
    clearBtn.addEventListener('click', () => {
        abstractInput.value = '';
        updateAbstractCounters();
        hideAbstractError();
        resultsEmpty.style.display = 'block';
        resultsContent.style.display = 'none';
        const absActionBtns = document.getElementById('abstract-action-buttons');
        if (absActionBtns) absActionBtns.style.display = 'none';
        if (classifyLangPill) classifyLangPill.style.display = 'none';
        if (resMultilingualBanner) resMultilingualBanner.style.display = 'none';
        liveBadge.textContent = 'READY';
        liveBadge.classList.remove('live');
        abstractInput.focus();
    });

    // Classify Abstract
    classifyBtn.addEventListener('click', async () => {
        const text = abstractInput.value.trim();
        if (!text) {
            showAbstractError('Please paste or type a medical research abstract before classifying.');
            return;
        }

        const words = text.split(/\s+/);
        if (words.length < 2) {
            showAbstractError('Input text is too short. Please enter at least 2 words.');
            return;
        }

        hideAbstractError();

        classifyBtn.disabled = true;
        btnText.textContent = 'Analyzing Abstract...';
        btnSpinner.style.display = 'inline-block';
        liveBadge.textContent = 'PROCESSING';
        liveBadge.classList.add('live');

        try {
            const modelType = modelSelector.value;
            const response = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, model_type: modelType })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Classification failed.');

            resultsEmpty.style.display = 'none';
            resultsContent.style.display = 'block';
            liveBadge.textContent = 'PREDICTED';
            liveBadge.classList.add('live');

            if (classifyLangPill && data.detected_language) {
                classifyLangPill.style.display = 'inline-flex';
                classifyLangPill.textContent = `🌐 ${data.detected_language.language_name.split('(')[0].trim()}`;
            }

            if (resMultilingualBanner) {
                if (data.is_translated && data.translated_clinical_query) {
                    resMultilingualBanner.style.display = 'block';
                    resDetectedLangTag.textContent = `🌐 Detected: ${data.detected_language.language_name}`;
                    resTransStatus.textContent = `${data.substitutions_made} medical terms normalized`;
                    resTransQuery.innerHTML = `<strong>Translated for NLP Classifier:</strong> <em>"${data.translated_clinical_query}"</em>`;
                } else {
                    resMultilingualBanner.style.display = 'none';
                }
            }

            // Biomedical Domain Guardrail (OOD Alert)
            const oodWarning = document.getElementById('abstract-ood-warning');
            const oodMsg = document.getElementById('abstract-ood-msg');
            if (data.warning) {
                if (oodWarning && oodMsg) {
                    oodMsg.textContent = data.warning;
                    oodWarning.style.display = 'flex';
                }
                showToast(data.warning, 'warning', 5000);
            } else {
                if (oodWarning) oodWarning.style.display = 'none';
                showToast(`Classification complete: ${data.predicted_category}`, 'success');
            }

            const absActionBtns = document.getElementById('abstract-action-buttons');
            if (absActionBtns) absActionBtns.style.display = 'inline-flex';
            resCategoryName.textContent = `[${data.predicted_class_id}] ${data.predicted_category}`;
            resCategoryDesc.textContent = data.category_description;
            resCategoryId.textContent = data.predicted_class_id;
            resModelName.textContent = data.model_used;
            resTokenCount.textContent = data.metrics.cleaned_word_count;

            // On mobile / tablet screens, smoothly auto-scroll down to classification probabilities
            if (window.innerWidth < 1024) {
                setTimeout(() => {
                    const resCard = document.getElementById('results-content');
                    if (resCard) {
                        resCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                }, 100);
            }

            probBarsList.innerHTML = '';
            if (data.probabilities && data.probabilities.length > 0) {
                data.probabilities.forEach(item => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = `prob-item ${item.is_top ? 'is-top-winner' : ''}`;

                    itemDiv.innerHTML = `
                        <div class="prob-row">
                            <span class="prob-name">
                                ${item.is_top ? '★ ' : ''}[${item.class_id}] ${item.category_name}
                            </span>
                            <span class="prob-val">${item.confidence_percent}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: 0%"></div>
                        </div>
                    `;

                    probBarsList.appendChild(itemDiv);

                    setTimeout(() => {
                        const fill = itemDiv.querySelector('.prob-bar-fill');
                        if (fill) fill.style.width = `${Math.max(item.confidence_percent, 1.5)}%`;
                    }, 50);
                });
            }

        } catch (err) {
            showAbstractError(err.message || 'Error communicating with classifier API.');
            liveBadge.textContent = 'ERROR';
            liveBadge.classList.remove('live');
        } finally {
            classifyBtn.disabled = false;
            btnText.textContent = 'Classify Abstract';
            btnSpinner.style.display = 'none';
        }
    });
});
