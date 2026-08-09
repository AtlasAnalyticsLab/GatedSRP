(() => {
  const menuButton = document.querySelector('.menu-button');
  const siteNav = document.querySelector('.site-nav');

  if (menuButton && siteNav) {
    const closeMenu = () => {
      menuButton.setAttribute('aria-expanded', 'false');
      siteNav.classList.remove('is-open');
    };

    menuButton.addEventListener('click', () => {
      const open = menuButton.getAttribute('aria-expanded') === 'true';
      menuButton.setAttribute('aria-expanded', String(!open));
      siteNav.classList.toggle('is-open', !open);
    });

    siteNav.addEventListener('click', (event) => {
      if (event.target instanceof HTMLAnchorElement) {
        closeMenu();
      }
    });

    // Resizing or rotating a device can cross the navigation breakpoint. Reset
    // the temporary menu state so it cannot reappear unexpectedly after a later
    // orientation change.
    let viewportWidth = window.innerWidth;
    window.addEventListener('resize', () => {
      if (window.innerWidth === viewportWidth) return;
      viewportWidth = window.innerWidth;
      closeMenu();
    });

    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || menuButton.getAttribute('aria-expanded') !== 'true') return;
      closeMenu();
      menuButton.focus();
    });
  }

  document.querySelectorAll('[data-tabs]').forEach((tabs) => {
    const tabButtons = Array.from(tabs.querySelectorAll('[role="tab"]'));
    const panels = Array.from(tabs.querySelectorAll('[role="tabpanel"]'));

    // Keep ARIA selection, keyboard focus, and panel visibility in one update
    // so mouse and keyboard interactions cannot drift into different states.
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
        // Clipboard access may be unavailable on non-secure local origins.
        // The source remains visible, so tell the reader to select it directly.
        button.textContent = 'Select text';
      }
    });
  });

  const year = String(new Date().getFullYear());
  document.querySelectorAll('[data-year]').forEach((node) => { node.textContent = year; });

  if ('IntersectionObserver' in window) {
    const navLinks = Array.from(document.querySelectorAll('.site-nav a[href^="#"]'));
    const sections = navLinks
      .map((link) => document.querySelector(link.getAttribute('href')))
      .filter(Boolean);
    // Use the section occupying the reading band, rather than raw scroll
    // position, to avoid rapidly switching links around section boundaries.
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      navLinks.forEach((link) => {
        const active = link.getAttribute('href') === `#${visible.target.id}`;
        if (active) link.setAttribute('aria-current', 'true');
        else link.removeAttribute('aria-current');
      });
    }, { rootMargin: '-25% 0px -60% 0px', threshold: [0, 0.2, 0.6] });
    sections.forEach((section) => observer.observe(section));
  }
})();
