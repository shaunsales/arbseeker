/**
 * Widget initialization functions for Strategy Lab.
 * Called after HTMX swaps to initialize TradingView Lightweight Charts.
 */

function initDataCharts() {
    const el = document.getElementById('ohlcv-chart-data');
    if (!el) return;

    const data = JSON.parse(el.textContent);
    if (!data || !data.timestamps || data.timestamps.length === 0) return;

    // Clear existing charts
    const priceEl = document.getElementById('price-chart');
    const volumeEl = document.getElementById('volume-chart');
    if (!priceEl || !volumeEl) return;
    priceEl.innerHTML = '';
    volumeEl.innerHTML = '';

    const chartOptions = {
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#9ca3af',
        },
        grid: {
            vertLines: { color: '#374151' },
            horzLines: { color: '#374151' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#374151' },
        timeScale: { borderColor: '#374151', timeVisible: true },
    };

    // Price chart
    const priceChart = LightweightCharts.createChart(priceEl, { ...chartOptions, height: 400 });
    const lineSeries = priceChart.addLineSeries({ color: '#3b82f6', lineWidth: 2 });
    const priceData = data.timestamps.map((t, i) => ({
        time: Math.floor(new Date(t).getTime() / 1000),
        value: data.close[i],
    }));
    lineSeries.setData(priceData);

    // Volume chart
    const volumeChart = LightweightCharts.createChart(volumeEl, { ...chartOptions, height: 120 });
    const volumeSeries = volumeChart.addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: '',
    });
    const volumeData = data.timestamps.map((t, i) => ({
        time: Math.floor(new Date(t).getTime() / 1000),
        value: data.volume[i],
        color: i === 0 ? '#6b7280' : (data.close[i] >= data.close[i-1] ? '#22c55e' : '#ef4444'),
    }));
    volumeSeries.setData(volumeData);

    // Sync time scales
    priceChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range) volumeChart.timeScale().setVisibleLogicalRange(range);
    });
    volumeChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range) priceChart.timeScale().setVisibleLogicalRange(range);
    });

    // Fit content
    priceChart.timeScale().setVisibleRange({
        from: priceData[0].time,
        to: priceData[priceData.length - 1].time,
    });
    volumeChart.timeScale().setVisibleRange({
        from: volumeData[0].time,
        to: volumeData[volumeData.length - 1].time,
    });
}


function initBasisCharts(quoteVenues) {
    const el = document.getElementById('basis-chart-data');
    if (!el) return;

    const data = JSON.parse(el.textContent);
    if (!data || !data.base_price || data.base_price.length === 0) return;

    const basisEl = document.getElementById('basis-chart');
    const priceEl = document.getElementById('price-chart');
    if (!basisEl || !priceEl) return;
    basisEl.innerHTML = '';
    priceEl.innerHTML = '';

    const chartOptions = {
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#9ca3af',
        },
        grid: {
            vertLines: { color: '#374151' },
            horzLines: { color: '#374151' },
        },
        rightPriceScale: { borderColor: '#374151' },
        timeScale: { borderColor: '#374151', timeVisible: true },
    };

    // Basis chart (bps)
    const basisChart = LightweightCharts.createChart(basisEl, { ...chartOptions, height: 300 });

    const venues = JSON.parse(document.getElementById('basis-venues-data').textContent);
    const colors = ['#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4'];

    venues.forEach((venue, idx) => {
        const basisKey = venue + '_basis_bps';
        if (!data[basisKey] || data[basisKey].length === 0) return;
        const series = basisChart.addLineSeries({
            color: colors[idx % colors.length],
            lineWidth: 1.5,
            title: venue + ' basis (bps)',
        });
        // Data is already [{time, value}, ...] with NaN filtered out
        series.setData(data[basisKey]);
    });

    // Zero line using first/last timestamps from basis data
    const firstVenueKey = venues[0] + '_basis_bps';
    if (data[firstVenueKey] && data[firstVenueKey].length > 1) {
        const zeroLine = basisChart.addLineSeries({ color: '#6b7280', lineWidth: 1, lineStyle: 2 });
        const first = data[firstVenueKey][0].time;
        const last = data[firstVenueKey][data[firstVenueKey].length - 1].time;
        zeroLine.setData([{ time: first, value: 0 }, { time: last, value: 0 }]);
    }

    // Price chart
    const priceChart = LightweightCharts.createChart(priceEl, { ...chartOptions, height: 200 });

    const baseLine = priceChart.addLineSeries({ color: '#3b82f6', lineWidth: 1.5, title: 'Base' });
    baseLine.setData(data.base_price);

    venues.forEach((venue, idx) => {
        const priceKey = venue + '_price';
        if (!data[priceKey] || data[priceKey].length === 0) return;
        const series = priceChart.addLineSeries({
            color: '#22c55e',
            lineWidth: 1.5,
            title: venue,
        });
        series.setData(data[priceKey]);
    });

    // Quality timeline chart
    const qualityEl = document.getElementById('quality-chart');
    let qualityChart = null;
    if (qualityEl && data.quality && data.quality.length > 0) {
        qualityEl.innerHTML = '';
        qualityChart = LightweightCharts.createChart(qualityEl, {
            ...chartOptions,
            height: 80,
            rightPriceScale: { visible: false },
            leftPriceScale: { visible: false },
        });
        const qualitySeries = qualityChart.addHistogramSeries({
            priceScaleId: '',
            priceFormat: { type: 'volume' },
        });
        qualitySeries.setData(data.quality);
    }

    // Sync all time scales
    const charts = [basisChart, priceChart, qualityChart].filter(Boolean);
    charts.forEach((src, si) => {
        src.timeScale().subscribeVisibleLogicalRangeChange(range => {
            if (!range) return;
            charts.forEach((dst, di) => {
                if (si !== di) dst.timeScale().setVisibleLogicalRange(range);
            });
        });
    });

    basisChart.timeScale().fitContent();
    priceChart.timeScale().fitContent();
    if (qualityChart) qualityChart.timeScale().fitContent();
}
