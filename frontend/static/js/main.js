document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file-upload');
    const loadingState = document.getElementById('loadingState');
    const errorState = document.getElementById('errorState');
    const errorMessage = document.getElementById('errorMessage');
    const segmentsContainer = document.getElementById('segmentsContainer');
    const segmentsGrid = document.getElementById('segmentsGrid');
    const totalUsers = document.getElementById('totalUsers');

    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            showError('Please select a file to upload');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            showLoading();
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to process file');
            }

            const data = await response.json();
            displaySegments(data);
        } catch (error) {
            showError(error.message);
        }
    });

    // Handle drag and drop
    const dropZone = document.querySelector('label[for="file-upload"]');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('bg-gray-200');
    }

    function unhighlight(e) {
        dropZone.classList.remove('bg-gray-200');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
    }

    function showLoading() {
        loadingState.classList.remove('hidden');
        errorState.classList.add('hidden');
        segmentsContainer.classList.add('hidden');
    }

    function showError(message) {
        loadingState.classList.add('hidden');
        errorState.classList.remove('hidden');
        errorMessage.textContent = message;
        segmentsContainer.classList.add('hidden');
    }

    function displaySegments(data) {
        loadingState.classList.add('hidden');
        errorState.classList.add('hidden');
        segmentsContainer.classList.remove('hidden');

        // Update total users
        totalUsers.textContent = data.total_users;

        // Clear existing segments
        segmentsGrid.innerHTML = '';

        // Create segment cards
        data.segments.forEach(segment => {
            const card = createSegmentCard(segment);
            segmentsGrid.appendChild(card);
        });
    }

    function createSegmentCard(segment) {
        const template = document.getElementById('segmentCardTemplate');
        const card = template.content.cloneNode(true);

        // Set segment name and user count
        card.querySelector('.segment-name').textContent = segment.name;
        card.querySelector('.user-count').textContent = `${segment.user_count} users`;

        // Set metrics
        card.querySelector('.total-actions').textContent = segment.metrics.total_actions;
        card.querySelector('.avg-revisions').textContent = segment.metrics.avg_revisions;
        card.querySelector('.avg-time-to-close').textContent = `${segment.metrics.avg_time_to_close}h`;

        // Set characteristics
        const characteristicsList = card.querySelector('.characteristics-list');
        segment.characteristics.forEach(char => {
            const li = document.createElement('li');
            li.textContent = char;
            characteristicsList.appendChild(li);
        });

        // Set pain points
        const painPointsList = card.querySelector('.pain-points-list');
        segment.pain_points.forEach(point => {
            const li = document.createElement('li');
            li.textContent = `${point.name} (${point.count} users)`;
            painPointsList.appendChild(li);
        });

        // Set recommendations
        const recommendationsList = card.querySelector('.recommendations-list');
        segment.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });

        return card;
    }
}); 