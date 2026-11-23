
(() => {
  const storageKey = "zerodefect.text-size";
  const sizes = [80, 90, 100, 110, 120, 130, 140, 150];
  const decrease = document.getElementById("text-size-decrease");
  const reset = document.getElementById("text-size-reset");
  const increase = document.getElementById("text-size-increase");
  const status = document.getElementById("text-size-status");
  let size = 100;

  try {
    const savedSize = Number(localStorage.getItem(storageKey));
    if (sizes.includes(savedSize)) size = savedSize;
  } catch {
    // Keep the default size when storage is unavailable.
  }

  function applySize() {
    document.documentElement.style.setProperty("--text-scale", size / 100);
    reset.textContent = `${size}%`;
    reset.setAttribute("aria-label", `기본 크기로 재설정, 현재 ${size}%`);
    decrease.disabled = size === sizes[0];
    increase.disabled = size === sizes[sizes.length - 1];
  }

  function changeSize(nextSize) {
    if (!sizes.includes(nextSize) || nextSize === size) return;
    size = nextSize;
    applySize();
    status.textContent = `글자 크기 ${size}%`;
    try {
      localStorage.setItem(storageKey, String(size));
    } catch {
      // The current page still supports resizing without storage.
    }
  }

  decrease.addEventListener("click", () => changeSize(sizes[sizes.indexOf(size) - 1]));
  reset.addEventListener("click", () => changeSize(100));
  increase.addEventListener("click", () => changeSize(sizes[sizes.indexOf(size) + 1]));
  applySize();
})();
