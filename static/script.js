document.addEventListener("DOMContentLoaded", () => {
    // Navigation Tabs
    const tabButtons = document.querySelectorAll(".tab-btn");
    const inputContainers = document.querySelectorAll(".tab-inputs");
    const panels = document.querySelectorAll(".tab-panel");

    // Input elements
    const livingAreaSlider = document.getElementById("living-area");
    const livingAreaVal = document.getElementById("living-area-val");
    const bedroomRadios = document.getElementsByName("bedrooms");
    const activeModelSelect = document.getElementById("active-model");
    
    // Financial Inputs
    const downPaymentSlider = document.getElementById("down-payment");
    const downPaymentVal = document.getElementById("down-payment-val");
    const interestRateSlider = document.getElementById("interest-rate");
    const interestRateVal = document.getElementById("interest-rate-val");
    const loanTermSelect = document.getElementById("loan-term");

    // Steppers
    const fullBathVal = document.getElementById("full-bath");
    const halfBathVal = document.getElementById("half-bath");
    const bsmtFullVal = document.getElementById("bsmt-full-bath");
    const bsmtHalfVal = document.getElementById("bsmt-half-bath");
    
    // Outputs
    const predictedPriceEl = document.getElementById("predicted-price");
    const pricePerSqftEl = document.getElementById("price-per-sqft");
    const totalBathsEl = document.getElementById("total-bathrooms-val");
    const marketStandingEl = document.getElementById("market-standing");
    const scaleFill = document.getElementById("visual-scale-fill");
    const scaleIndicator = document.getElementById("visual-scale-indicator");
    const activeModelNameEl = document.getElementById("result-model-name");

    // Financial Outputs
    const mortgagePaymentEl = document.getElementById("mortgage-payment");
    const mortgagePiEl = document.getElementById("mortgage-pi");
    const propertyTaxEl = document.getElementById("property-tax");
    const loanAmountEl = document.getElementById("loan-amount");
    const requiredIncomeEl = document.getElementById("required-income");

    // Tab 3 Outputs
    const metricsTableBody = document.getElementById("metrics-table-body");

    // State Variables
    let currentPrice = 0;
    let predictionsData = {};
    let metricsData = {};
    let coefData = {};
    let lastStats = {};
    let activeModel = "ElasticNet Regression";

    // Chart.js instances
    let modelChartInstance = null;
    let coefChartInstance = null;

    // ==========================================
    // 1. TAB NAVIGATION
    // ==========================================
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            // Toggle active tab buttons
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Toggle active input categories
            inputContainers.forEach(container => {
                container.classList.remove("active");
                if (container.id === targetTab.replace("tab-", "inputs-")) {
                    container.classList.add("active");
                }
            });

            // Toggle active result panels
            panels.forEach(panel => {
                panel.classList.remove("active");
                if (panel.id === targetTab.replace("tab-", "panel-")) {
                    panel.classList.add("active");
                }
            });

            // Trigger chart redraws if performance tab is opened
            if (targetTab === "tab-performance") {
                setTimeout(renderCharts, 50);
            }
        });
    });

    // ==========================================
    // 2. STATA ACQUISITION & INIT
    // ==========================================
    async function initStats() {
        try {
            const res = await fetch("/api/stats");
            if (res.ok) {
                lastStats = await res.json();
                livingAreaSlider.min = lastStats.grLivArea.min || 334;
                livingAreaSlider.max = lastStats.grLivArea.max || 4000;
                livingAreaSlider.value = Math.round(lastStats.grLivArea.mean) || 1500;
                livingAreaVal.textContent = Number(livingAreaSlider.value).toLocaleString() + " sq ft";
            }
        } catch (e) {
            console.warn("Failed to fetch dataset stats. Using fallbacks.", e);
        }
        
        // Fetch model coefficients for Chart.js
        try {
            const res = await fetch("/api/coefficients");
            if (res.ok) {
                const data = await res.json();
                coefData = data.coefficients;
            }
        } catch (e) {
            console.warn("Failed to load coefficients.", e);
        }
        
        updatePrediction();
    }

    // Stepper function handler
    window.adjustStepper = function(id, delta) {
        const valEl = document.getElementById(id);
        let val = parseInt(valEl.textContent) || 0;
        val = Math.max(0, Math.min(10, val + delta));
        valEl.textContent = val;
        updatePrediction();
    };

    function getSelectedBedrooms() {
        for (const radio of bedroomRadios) {
            if (radio.checked) {
                return parseInt(radio.value);
            }
        }
        return 3;
    }

    // ==========================================
    // 3. CORE CALCULATION & INFERENCE
    // ==========================================
    async function updatePrediction() {
        const grLivArea = parseInt(livingAreaSlider.value) || 1500;
        const bedroomAbvGr = getSelectedBedrooms();
        const fullBath = parseInt(fullBathVal.textContent) || 0;
        const halfBath = parseInt(halfBathVal.textContent) || 0;
        const bsmtFullBath = parseInt(bsmtFullVal.textContent) || 0;
        const bsmtHalfBath = parseInt(bsmtHalfVal.textContent) || 0;
        
        livingAreaVal.textContent = grLivArea.toLocaleString() + " sq ft";
        
        const totalBaths = fullBath + (0.5 * halfBath) + bsmtFullBath + (0.5 * bsmtHalfBath);
        totalBathsEl.textContent = totalBaths;

        const payload = {
            grLivArea,
            bedroomAbvGr,
            fullBath,
            halfBath,
            bsmtFullBath,
            bsmtHalfBath
        };

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const data = await response.json();
                predictionsData = data.predictions;
                metricsData = data.metrics;

                refreshUI();
            }
        } catch (err) {
            console.error("API connection failed.", err);
        }
    }

    // Update UI components when prediction payload changes
    function refreshUI() {
        // Read value based on selected active model
        activeModel = activeModelSelect.value;
        const predictedVal = Math.round(predictionsData[activeModel] || 150000);
        
        // Update model header label
        activeModelNameEl.textContent = `Estimated Value (${activeModel})`;

        // Animate price display counter
        animatePriceCounter(currentPrice, predictedVal, 300);
        currentPrice = predictedVal;

        // Market standing evaluation
        let marketText = "Near Average";
        if (predictedVal < 120000) {
            marketText = "Affordable Budget Home";
            marketStandingEl.className = "metric-val";
            marketStandingEl.style.color = "#38bdf8"; // Light Blue
        } else if (predictedVal > 280000) {
            marketText = "Premium Luxury Segment";
            marketStandingEl.className = "metric-val";
            marketStandingEl.style.color = "#10b981"; // Emerald
        } else {
            marketStandingEl.className = "metric-val highlight";
            marketStandingEl.style.color = "#6366f1"; // Indigo
        }
        marketStandingEl.textContent = marketText;

        // Visual Scale Progress Bar ($50,000 to $400,000)
        const minPrice = 50000;
        const maxPrice = 400000;
        let percent = ((predictedVal - minPrice) / (maxPrice - minPrice)) * 100;
        percent = Math.max(0, Math.min(100, percent));
        scaleFill.style.width = percent + "%";
        scaleIndicator.style.left = percent + "%";

        // Update financial estimates
        runFinancialProjections(predictedVal);

        // Update performance metrics table
        renderMetricsTable();

        // Update Chart JS figures if active
        renderCharts();
    }

    function animatePriceCounter(start, end, duration) {
        if (start === end) {
            predictedPriceEl.textContent = "$" + end.toLocaleString();
            updatePriceLabels(end);
            return;
        }
        const range = end - start;
        const startTime = new Date().getTime();
        const endTime = startTime + duration;
        
        function run() {
            const now = new Date().getTime();
            const remaining = Math.max((endTime - now) / duration, 0);
            const value = Math.round(end - (remaining * range));
            predictedPriceEl.textContent = "$" + value.toLocaleString();
            updatePriceLabels(value);
            
            if (value !== end) {
                requestAnimationFrame(run);
            }
        }
        requestAnimationFrame(run);
    }

    function updatePriceLabels(priceVal) {
        const sqft = parseInt(livingAreaSlider.value) || 1;
        const pricePerSqft = Math.round(priceVal / sqft);
        pricePerSqftEl.textContent = "$" + pricePerSqft.toLocaleString() + "/sq ft";
    }

    // ==========================================
    // 4. FINANCIAL CALCULATIONS
    // ==========================================
    function runFinancialProjections(housePrice) {
        const downPercent = parseInt(downPaymentSlider.value) || 20;
        const interestRate = parseFloat(interestRateSlider.value) || 6.5;
        const loanTermYears = parseInt(loanTermSelect.value) || 30;

        downPaymentVal.textContent = `${downPercent}%`;
        interestRateVal.textContent = `${interestRate.toFixed(1)}%`;

        const downPaymentAmount = housePrice * (downPercent / 100);
        const loanAmount = Math.max(0, housePrice - downPaymentAmount);

        // Loan formula components
        const monthlyRate = (interestRate / 12) / 100;
        const totalPayments = loanTermYears * 12;

        let monthlyPI = 0;
        if (loanAmount > 0) {
            if (monthlyRate === 0) {
                monthlyPI = loanAmount / totalPayments;
            } else {
                monthlyPI = loanAmount * (monthlyRate * Math.pow(1 + monthlyRate, totalPayments)) / (Math.pow(1 + monthlyRate, totalPayments) - 1);
            }
        }

        // Est Property Tax (1.2% average annually, divided monthly)
        const monthlyPropertyTax = (housePrice * 0.012) / 12;

        const totalMonthlyPayment = monthlyPI + monthlyPropertyTax;

        // Required income: Recommended maximum monthly housing cost is 28% of gross monthly income
        const recommendedMonthlyIncome = totalMonthlyPayment / 0.28;
        const recommendedAnnualIncome = recommendedMonthlyIncome * 12;

        // Render values
        mortgagePaymentEl.innerHTML = `$${Math.round(totalMonthlyPayment).toLocaleString()}<span>/mo</span>`;
        mortgagePiEl.textContent = `$${Math.round(monthlyPI).toLocaleString()}/mo`;
        propertyTaxEl.textContent = `$${Math.round(monthlyPropertyTax).toLocaleString()}/mo`;
        loanAmountEl.textContent = `$${Math.round(loanAmount).toLocaleString()}`;
        requiredIncomeEl.textContent = `$${Math.round(recommendedAnnualIncome).toLocaleString()}/yr`;
    }

    // ==========================================
    // 5. PERFORMANCE METRIC TABLES
    // ==========================================
    function renderMetricsTable() {
        if (!predictionsData || !metricsData) return;
        metricsTableBody.innerHTML = "";

        Object.keys(predictionsData).forEach(modelName => {
            const isSelected = modelName === activeModel;
            const predVal = predictionsData[modelName];
            const m = metricsData[modelName] || {};
            
            const tr = document.createElement("tr");
            if (isSelected) tr.style.background = "rgba(99, 102, 241, 0.12)";
            
            tr.innerHTML = `
                <td style="font-weight: ${isSelected ? '700' : '500'}; color: ${isSelected ? '#a5b4fc' : 'inherit'}">
                    ${isSelected ? '👉 ' : ''}${modelName}
                </td>
                <td class="${isSelected ? 'highlight' : ''}">$${Math.round(predVal).toLocaleString()}</td>
                <td>${m.R2 ? m.R2.toFixed(4) : "N/A"}</td>
                <td>$${m.MAE ? Math.round(m.MAE).toLocaleString() : "N/A"}</td>
                <td>${m.RMSLE ? m.RMSLE.toFixed(4) : "N/A"}</td>
            `;
            metricsTableBody.appendChild(tr);
        });
    }

    // ==========================================
    // 6. CHART.JS VISUALIZATIONS
    // ==========================================
    function renderCharts() {
        const comparisonTabActive = document.querySelector(".tab-btn[data-tab='tab-performance']").classList.contains("active");
        if (!comparisonTabActive || !predictionsData || Object.keys(predictionsData).length === 0) return;

        // Render prediction comparisons chart
        const modelCtx = document.getElementById("modelComparisonChart").getContext("2d");
        const modelNames = Object.keys(predictionsData);
        const predictedPrices = modelNames.map(name => Math.round(predictionsData[name]));

        if (modelChartInstance) {
            modelChartInstance.destroy();
        }

        modelChartInstance = new Chart(modelCtx, {
            type: 'bar',
            data: {
                labels: modelNames.map(name => name.replace(" Regression", "")),
                datasets: [{
                    label: 'Predicted Valuation ($)',
                    data: predictedPrices,
                    backgroundColor: modelNames.map(name => name === activeModel ? 'rgba(16, 185, 129, 0.6)' : 'rgba(99, 102, 241, 0.5)'),
                    borderColor: modelNames.map(name => name === activeModel ? '#10b981' : '#6366f1'),
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: {
                            color: '#94a3b8',
                            callback: value => '$' + (value / 1000) + 'k'
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8', font: { size: 9 } }
                    }
                }
            }
        });

        // Render Coefficient Impact weights chart (log scale ElasticNet weights)
        if (!coefData || Object.keys(coefData).length === 0) return;
        const coefCtx = document.getElementById("coefChart").getContext("2d");
        
        // ElasticNet is the tuned model
        const weights = coefData["ElasticNet Regression"] || [0, 0, 0];
        const labels = ["Living Area", "Bedrooms", "Total Baths"];

        if (coefChartInstance) {
            coefChartInstance.destroy();
        }

        coefChartInstance = new Chart(coefCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Weight Impact',
                    data: weights,
                    backgroundColor: weights.map(w => w >= 0 ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)'),
                    borderColor: weights.map(w => w >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }

    // ==========================================
    // 7. EXPORT DATA REPORT
    // ==========================================
    window.downloadJsonReport = function() {
        const report = {
            valuationTimestamp: new Date().toISOString(),
            valuationModel: activeModel,
            characteristics: {
                aboveGradeLivingAreaSqft: parseInt(livingAreaSlider.value),
                bedroomsAboveGrade: getSelectedBedrooms(),
                bathrooms: {
                    full: parseInt(fullBathVal.textContent),
                    half: parseInt(halfBathVal.textContent),
                    basementFull: parseInt(bsmtFullVal.textContent),
                    basementHalf: parseInt(bsmtHalfVal.textContent),
                    calculatedScore: parseFloat(totalBathsEl.textContent)
                }
            },
            predictionsByModel: predictionsData,
            financialEstimates: {
                loanTermYears: parseInt(loanTermSelect.value),
                downPaymentPercent: parseInt(downPaymentSlider.value),
                annualInterestRatePercent: parseFloat(interestRateSlider.value),
                estimatedMortgagePaymentMonthly: parseFloat(mortgagePaymentEl.textContent.replace(/[^0-9.-]+/g,"")),
                propertyTaxMonthly: parseFloat(propertyTaxEl.textContent.replace(/[^0-9.-]+/g,"")),
                loanAmount: parseFloat(loanAmountEl.textContent.replace(/[^0-9.-]+/g,"")),
                recommendedAnnualIncome: parseFloat(requiredIncomeEl.textContent.replace(/[^0-9.-]+/g,""))
            }
        };

        const blob = new Blob([JSON.stringify(report, null, 4)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `ValuAI_Estimation_Report_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    // ==========================================
    // 8. EVENT LISTENERS
    // ==========================================
    livingAreaSlider.addEventListener("input", () => {
        livingAreaVal.textContent = parseInt(livingAreaSlider.value).toLocaleString() + " sq ft";
    });
    livingAreaSlider.addEventListener("change", updatePrediction);
    
    for (const radio of bedroomRadios) {
        radio.addEventListener("change", updatePrediction);
    }
    
    activeModelSelect.addEventListener("change", refreshUI);

    downPaymentSlider.addEventListener("input", () => {
        downPaymentVal.textContent = `${downPaymentSlider.value}%`;
        runFinancialProjections(currentPrice);
    });
    
    interestRateSlider.addEventListener("input", () => {
        interestRateVal.textContent = `${parseFloat(interestRateSlider.value).toFixed(1)}%`;
        runFinancialProjections(currentPrice);
    });

    loanTermSelect.addEventListener("change", () => {
        runFinancialProjections(currentPrice);
    });
    
    // Initialize Dashboard
    initStats();
});
