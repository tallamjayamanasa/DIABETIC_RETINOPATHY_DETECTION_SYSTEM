/**
 * Diabetic Retinopathy Detection System - Main JavaScript
 * Handles drag-and-drop uploads, previews, dynamic UI, and Chart.js integrations.
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Navbar Scroll Shadow Effect
  const navbar = document.querySelector('.medical-navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 20) {
        navbar.classList.add('shadow-lg');
      } else {
        navbar.classList.remove('shadow-lg');
      }
    });
  }

  // 2. Drag & Drop Retinal Image Upload
  const dropzone = document.getElementById('retinalDropzone');
  const fileInput = document.getElementById('retinalImageInput');
  const previewContainer = document.getElementById('imagePreviewContainer');
  const previewImg = document.getElementById('imagePreview');
  const previewFileName = document.getElementById('previewFileName');
  const previewFileSize = document.getElementById('previewFileSize');
  const clearBtn = document.getElementById('clearImageBtn');
  const uploadForm = document.getElementById('retinalUploadForm');
  const submitBtn = document.getElementById('submitAnalyzeBtn');
  const uploadSpinner = document.getElementById('uploadSpinner');

  if (dropzone && fileInput) {
    // Click on dropzone triggers hidden file input
    dropzone.addEventListener('click', (e) => {
      if (e.target !== clearBtn && !clearBtn?.contains(e.target)) {
        fileInput.click();
      }
    });

    // Drag-over styling
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        fileInput.files = files;
        handleSelectedFile(files[0]);
      }
    });

    // Handle file selected via file dialog
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        handleSelectedFile(fileInput.files[0]);
      }
    });

    // Clear selected file
    if (clearBtn) {
      clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        if (previewContainer) previewContainer.style.display = 'none';
        if (previewImg) previewImg.src = '';
        if (submitBtn) submitBtn.disabled = true;
      });
    }

    // Submit handler with retinal laser animation
    if (uploadForm) {
      uploadForm.addEventListener('submit', function (e) {
        if (!fileInput.files || fileInput.files.length === 0) {
          e.preventDefault();
          alert('Please select or drag a retinal fundus image to analyze.');
          return;
        }

        if (submitBtn) {
          submitBtn.disabled = true;
          submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing Retinal Scan...';
        }
        if (previewContainer) {
          previewContainer.classList.add('scanning-active');
        }
      });
    }
  }

  function handleSelectedFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid retinal image in JPG, JPEG, or PNG format.');
      fileInput.value = '';
      return;
    }

    // Validate size (max 15MB)
    const maxSize = 15 * 1024 * 1024;
    if (file.size > maxSize) {
      alert(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 15 MB.`);
      fileInput.value = '';
      return;
    }

    // Render image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      if (previewFileName) previewFileName.textContent = file.name;
      if (previewFileSize) previewFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
      if (previewContainer) previewContainer.style.display = 'block';
      if (submitBtn) submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // 3. Auto-dismiss alerts after 6 seconds
  const autoAlerts = document.querySelectorAll('.alert-dismissible');
  autoAlerts.forEach((alertEl) => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alertEl);
      if (bsAlert) bsAlert.close();
    }, 6000);
  });

  // 4. Initialize Tooltips
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
});

/**
 * Render probability bar chart on prediction result page
 */
function renderProbabilityChart(canvasId, probabilitiesData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !probabilitiesData) return;

  const labels = Object.keys(probabilitiesData);
  const values = Object.values(probabilitiesData);

  // Map labels to full clinical format
  const displayLabels = {
    'No_DR': 'No Diabetic Retinopathy (Grade 0)',
    'Mild': 'Mild NPDR (Grade 1)',
    'Moderate': 'Moderate NPDR (Grade 2)',
    'Severe': 'Severe NPDR (Grade 3)',
    'Proliferative_DR': 'Proliferative DR (Grade 4)'
  };

  const stageColors = {
    'No_DR': '#059669',
    'Mild': '#0284c7',
    'Moderate': '#d97706',
    'Severe': '#ea580c',
    'Proliferative_DR': '#dc2626'
  };

  const formattedLabels = labels.map(l => displayLabels[l] || l);
  const backgroundColors = labels.map(l => stageColors[l] || '#0284c7');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: formattedLabels,
      datasets: [{
        label: 'Confidence Probability (%)',
        data: values,
        backgroundColor: backgroundColors,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.4)'
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (context) {
              return ` Probability: ${context.parsed.x.toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        x: {
          min: 0,
          max: 100,
          ticks: {
            callback: value => `${value}%`
          },
          grid: { color: 'rgba(0, 0, 0, 0.05)' }
        },
        y: {
          grid: { display: false }
        }
      }
    }
  });
}
