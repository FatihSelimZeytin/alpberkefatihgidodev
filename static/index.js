const studyAreaBounds = {
  minLon: 31.0,
  minLat: 36.8,
  maxLon: 34.8,
  maxLat: 39.3
};

const featureDescriptions = {
  B2_Blue: {
    title: 'B2 Blue',
    body: 'Sentinel-2 blue band. It helps distinguish surface brightness, bare soil, water, and atmospheric effects.'
  },
  B3_Green: {
    title: 'B3 Green',
    body: 'Sentinel-2 green band. It represents visible reflectance from vegetation and supports agricultural land detection.'
  },
  B4_Red: {
    title: 'B4 Red',
    body: 'Sentinel-2 red band. Plants absorb red light for photosynthesis, so this band is important for vegetation analysis.'
  },
  B8_NIR: {
    title: 'B8 NIR',
    body: 'Near-infrared band. Healthy vegetation strongly reflects NIR, making it one of the strongest crop condition indicators.'
  },
  B11_SWIR1: {
    title: 'B11 SWIR1',
    body: 'Short-wave infrared band. It provides information about soil moisture, vegetation moisture, drought, and water stress.'
  },
  B12_SWIR2: {
    title: 'B12 SWIR2',
    body: 'Second short-wave infrared band. It helps identify dry surfaces, bare soil, and moisture differences.'
  },
  NDVI: {
    title: 'NDVI',
    body: 'Normalized Difference Vegetation Index. It measures vegetation density and health; higher values usually indicate stronger vegetation.'
  },
  NDWI: {
    title: 'NDWI',
    body: 'Normalized Difference Water Index. It is related to surface and vegetation moisture, which is important for agricultural suitability.'
  },
  EVI: {
    title: 'EVI',
    body: 'Enhanced Vegetation Index. Similar to NDVI, but often more stable in dense vegetation and under atmospheric effects.'
  },
  SAVI: {
    title: 'SAVI',
    body: 'Soil Adjusted Vegetation Index. It reduces soil background effects, especially where vegetation cover is sparse.'
  },
  Elevation: {
    title: 'Elevation',
    body: 'Elevation above sea level. It affects temperature, rainfall, soil conditions, and crop growing suitability.'
  },
  Slope: {
    title: 'Slope',
    body: 'Terrain steepness. Lower slopes are generally more suitable for agriculture because they reduce erosion and irrigation difficulty.'
  },
  Aspect: {
    title: 'Aspect',
    body: 'Terrain facing direction. It can affect sunlight exposure, local moisture, and microclimate conditions.'
  },
  Annual_Rainfall: {
    title: 'Annual Rainfall',
    body: 'Total annual precipitation. It is a key climate factor for crop growth, especially in drought-sensitive regions such as Konya.'
  },
  LST_Celsius: {
    title: 'LST Celsius',
    body: 'Land Surface Temperature in Celsius. High surface temperature can indicate drought, heat stress, and reduced crop suitability.'
  }
};

const map = L.map('map').setView([38.1, 32.8], 8);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap',
  maxZoom: 18
}).addTo(map);

const studyAreaLayer = L.rectangle(
  [
    [studyAreaBounds.minLat, studyAreaBounds.minLon],
    [studyAreaBounds.maxLat, studyAreaBounds.maxLon]
  ],
  {
    color: '#1D9E75',
    weight: 2,
    fillColor: '#1D9E75',
    fillOpacity: 0.06
  }
).addTo(map);

studyAreaLayer.bindTooltip('Study area: Konya and surroundings');
map.fitBounds(studyAreaLayer.getBounds());

let marker = null;
let predictionLayer = null;

function colorForProbability(probability) {
  if (probability >= 0.7) return '#1D9E75';
  if (probability >= 0.4) return '#F2B84B';
  return '#E24B4A';
}

function labelForProbability(probability) {
  if (probability >= 0.7) return 'High suitability';
  if (probability >= 0.4) return 'Medium suitability';
  return 'Low suitability';
}

async function loadPredictionLayer() {
  try {
    const res = await fetch('/static/predictions.geojson', { cache: 'no-store' });
    if (!res.ok) return;

    const geojson = await res.json();
    predictionLayer = L.geoJSON(geojson, {
      pointToLayer: function (feature, latlng) {
        const probability = Number(feature.properties.probability);
        return L.circleMarker(latlng, {
          radius: 5,
          color: colorForProbability(probability),
          fillColor: colorForProbability(probability),
          fillOpacity: 0.62,
          weight: 1
        });
      },
      onEachFeature: function (feature, layer) {
        const probability = Math.round(Number(feature.properties.probability) * 100);
        layer.bindPopup(`
          <strong>${labelForProbability(feature.properties.probability)}</strong><br>
          Suitability probability: ${probability}%
        `);
      }
    });

    L.control.layers(null, {
      'Suitability prediction grid': predictionLayer,
      'Study area': studyAreaLayer
    }, { collapsed: false }).addTo(map);
  } catch (err) {
    // The grid layer is optional until the GEE prediction CSV is exported and processed.
  }
}

loadPredictionLayer();

function ensureResultPanel() {
  let panel = document.getElementById('result-panel');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'result-panel';
    document.getElementById('controls').after(panel);
  }
  return panel;
}

function ensureFeatureModal() {
  let modal = document.getElementById('feature-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'feature-modal';
    modal.innerHTML = `
      <div class="feature-modal-backdrop" data-close-feature-modal></div>
      <div class="feature-modal-card" role="dialog" aria-modal="true" aria-labelledby="feature-modal-title">
        <button class="feature-modal-close" type="button" data-close-feature-modal aria-label="Close">x</button>
        <h2 id="feature-modal-title"></h2>
        <p id="feature-modal-body"></p>
      </div>
    `;
    document.body.appendChild(modal);
  }
  return modal;
}

function openFeatureModal(key) {
  const info = featureDescriptions[key];
  if (!info) return;

  const modal = ensureFeatureModal();
  modal.querySelector('#feature-modal-title').textContent = info.title;
  modal.querySelector('#feature-modal-body').textContent = info.body;
  modal.classList.add('is-open');
}

function closeFeatureModal() {
  const modal = document.getElementById('feature-modal');
  if (modal) modal.classList.remove('is-open');
}

document.addEventListener('click', function (event) {
  const featureButton = event.target.closest('[data-feature-key]');
  if (featureButton) {
    openFeatureModal(featureButton.dataset.featureKey);
    return;
  }

  if (event.target.closest('[data-close-feature-modal]')) {
    closeFeatureModal();
  }
});

document.addEventListener('keydown', function (event) {
  if (event.key === 'Escape') {
    closeFeatureModal();
  }
});

function buildPayload(lat, lng) {
  return {
    Latitude: lat,
    Longitude: lng
  };
}

function featureLine(features) {
  const fields = [
    ['B2_Blue', features.B2_Blue],
    ['B3_Green', features.B3_Green],
    ['B4_Red', features.B4_Red],
    ['B8_NIR', features.B8_NIR],
    ['B11_SWIR1', features.B11_SWIR1],
    ['B12_SWIR2', features.B12_SWIR2],
    ['NDVI', features.NDVI],
    ['NDWI', features.NDWI],
    ['EVI', features.EVI],
    ['SAVI', features.SAVI],
    ['Elevation', features.Elevation],
    ['Slope', features.Slope],
    ['Aspect', features.Aspect],
    ['Annual_Rainfall', features.Annual_Rainfall],
    ['LST_Celsius', features.LST_Celsius]
  ];

  return fields
    .map(([key, value]) => {
      const title = featureDescriptions[key].title;
      return `
        <button class="feature-chip" type="button" data-feature-key="${key}">
          <strong>${title}:</strong> ${Number(value).toFixed(2)}
        </button>
      `;
    })
    .join('');
}

function explanationBlock(explanation) {
  if (!explanation || !Array.isArray(explanation.reasons) || explanation.reasons.length === 0) {
    return '';
  }

  return `
    <div class="explanation-box">
      <h3>Why this result?</h3>
      <p>${explanation.summary}</p>
      <ul>
        ${explanation.reasons.map((reason) => `
          <li>
            <strong>${reason.title}:</strong> ${reason.text}
          </li>
        `).join('')}
      </ul>
    </div>
  `;
}

function showResult(data) {
  const panel = ensureResultPanel();
  const pct = data.probability === null ? null : Math.round(data.probability * 100);
  const label = data.prediction === 1 ? 'Suitable' : 'Not suitable';
  const color = data.prediction === 1 ? '#1D9E75' : '#E24B4A';
  const nearest = data.nearest_point;
  const sourceText = data.source === 'prediction_grid'
    ? 'Prediction grid point'
    : 'Nearest sample point';
  const labelText = nearest.label === null || nearest.label === undefined
    ? ''
    : ` · Reference label: ${nearest.label}`;

  panel.innerHTML = `
    <div class="result-card">
      <div class="result-meta">Konya and surroundings · ${data.best_model}</div>
      <div class="result-title" style="color:${color}">${label}</div>
      ${
        pct === null
          ? ''
          : `<div class="bar"><div style="width:${pct}%;background:${color}"></div></div>
             <div class="result-small">Suitability probability: <strong>${pct}%</strong></div>`
      }
      <div class="result-small">
        ${sourceText}: ${nearest.latitude}, ${nearest.longitude}${labelText}
      </div>
      ${explanationBlock(data.explanation)}
      <div class="result-features">${featureLine(data.features)}</div>
    </div>
  `;

  if (marker) {
    marker.bindPopup(`<strong>${label}</strong>${pct === null ? '' : `<br>${pct}%`}`).openPopup();
  }
}

function showError(msg) {
  const panel = ensureResultPanel();
  panel.innerHTML = `<div class="error">${msg}</div>`;
}

map.on('click', async function (e) {
  const { lat, lng } = e.latlng;

  if (marker) marker.remove();
  marker = L.marker([lat, lng]).addTo(map);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload(lat, lng))
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Server error');
      marker.bindPopup('Outside the study area').openPopup();
      return;
    }

    showResult(data);
  } catch (err) {
    showError('Could not connect to the Flask server. Is python app.py running in the project environment?');
  }
});
