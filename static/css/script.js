// script.js
document.getElementById('submitForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    // Start progress bar animation
    var progressBar = document.getElementById('progressBar');
    progressBar.innerHTML = '<div class="progress"></div>';
    var progress = 0;
    var interval = setInterval(function() {
        progress += 1;
        if (progress >= 100) {
            clearInterval(interval);
            // Submit form data using AJAX or fetch
            var formData = new FormData(document.getElementById('submitForm'));
            fetch('{{ url_for("predict_datapoint") }}', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (response.ok) {
                    // If response is successful, redirect to the result page
                    window.location.href = '{{ url_for("loader") }}';
                } else {
                    console.error('Error:', response.statusText);
                    // Handle error here
                }
            })
            .catch(error => {
                console.error('Error:', error);
                // Handle error here
            });
        }
        document.querySelector('.progress').style.width = progress + '%';
    }, 50);
});
