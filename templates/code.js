async function uploadPhoto(file) {
  errorBox.style.display = 'none';
  dropzone.style.display = 'none';
  loader.style.display = 'flex';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      body: formData,
    });

    // Handle non-200 responses safely without crashing on .json()
    if (!response.ok) {
      let errorText = `Server Error (${response.status})`;
      try {
        const errorJson = await response.json();
        errorText = errorJson.detail || errorText;
      } catch {
        const raw = await response.text();
        if (raw) errorText = raw.slice(0, 150);
      }
      throw new Error(errorText);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.style.display = 'block';
    dropzone.style.display = 'block';
  } finally {
    loader.style.display = 'none';
  }
}