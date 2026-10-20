// Prepix Documentation Website Client Interactions
document.addEventListener('DOMContentLoaded', () => {
  // 1. Copy to Clipboard functionality
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      let textToCopy = '';

      if (targetId) {
        const el = document.getElementById(targetId);
        textToCopy = el ? el.innerText : '';
      } else {
        const codeSibling = btn.closest('.panel-header')
          ? btn.closest('.code-panel').querySelector('pre.active, pre.code-block')
          : null;
        textToCopy = codeSibling ? codeSibling.innerText : btn.parentElement.innerText.trim();
      }

      if (textToCopy) {
        navigator.clipboard.writeText(textToCopy).then(() => {
          const originalHTML = btn.innerHTML;
          btn.innerHTML = '<span>✓ Copied!</span>';
          setTimeout(() => {
            btn.innerHTML = originalHTML;
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy text: ', err);
        });
      }
    });
  });

  // 2. Tab switching logic for code panels
  document.querySelectorAll('.code-panel').forEach(panel => {
    const tabs = panel.querySelectorAll('.tab-btn');
    const blocks = panel.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const targetTab = tab.getAttribute('data-tab');

        tabs.forEach(t => t.classList.remove('active'));
        blocks.forEach(b => b.style.display = 'none');

        tab.classList.add('active');
        const activeBlock = panel.querySelector(`.tab-content[data-content="${targetTab}"]`);
        if (activeBlock) {
          activeBlock.style.display = 'block';
        }
      });
    });
  });

  // 3. Active sidebar link tracking during scroll
  const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -70% 0px',
    threshold: 0
  };

  const sections = document.querySelectorAll('.function-doc-item, section[id]');
  const navLinks = document.querySelectorAll('.sidebar-menu a, nav.main-nav a');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        navLinks.forEach(link => {
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('active');
          } else if (link.getAttribute('href')?.startsWith('#')) {
            link.classList.remove('active');
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(sec => observer.observe(sec));
});
