(() => {
  document.querySelectorAll('[data-tabs]').forEach((tabs) => {
    const tabButtons = Array.from(tabs.querySelectorAll('[role="tab"]'));
    const panels = Array.from(tabs.querySelectorAll('[role="tabpanel"]'));

    // Keep selection, keyboard focus, and panel visibility in one update so
    // pointer and keyboard interactions always expose the same result view.
    const activate = (nextTab) => {
      tabButtons.forEach((tab) => {
        const selected = tab === nextTab;
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
      });
      panels.forEach((panel) => {
        panel.hidden = panel.id !== nextTab.getAttribute('aria-controls');
      });
    };

    tabButtons.forEach((tab, index) => {
      tab.addEventListener('click', () => activate(tab));
      tab.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();

        let nextIndex = index;
        if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabButtons.length) % tabButtons.length;
        if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabButtons.length;
        if (event.key === 'Home') nextIndex = 0;
        if (event.key === 'End') nextIndex = tabButtons.length - 1;

        activate(tabButtons[nextIndex]);
        tabButtons[nextIndex].focus();
      });
    });
  });

  document.querySelectorAll('[data-copy-target]').forEach((button) => {
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;

      try {
        await navigator.clipboard.writeText(target.innerText);
        const original = button.textContent;
        button.textContent = 'Copied';
        window.setTimeout(() => { button.textContent = original; }, 1600);
      } catch {
        // Clipboard access can be unavailable on local or restricted origins;
        // the visible source remains selectable as a reliable fallback.
        button.textContent = 'Select text';
      }
    });
  });

  const year = String(new Date().getFullYear());
  document.querySelectorAll('[data-year]').forEach((node) => {
    node.textContent = year;
  });
})();
