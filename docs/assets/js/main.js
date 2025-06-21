// Main JavaScript for Imhotep Documentation Site

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all functionality
    initNavigation();
    initSmoothScrolling();
    initCodeCopyButtons();
    initTableOfContents();
    initSearchFunctionality();
    initThemeToggle();
    initAnimations();
});

// Navigation functionality
function initNavigation() {
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    const navTrigger = document.querySelector('#nav-trigger');
    
    // Mobile navigation toggle
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navTrigger.checked = !navTrigger.checked;
        });
        
        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!navMenu.contains(event.target) && !navToggle.contains(event.target)) {
                navTrigger.checked = false;
            }
        });
        
        // Close mobile menu when clicking on a link
        const navLinks = navMenu.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                navTrigger.checked = false;
            });
        });
    }
    
    // Dropdown navigation
    const dropdowns = document.querySelectorAll('.nav-dropdown');
    dropdowns.forEach(dropdown => {
        const toggle = dropdown.querySelector('.dropdown-toggle');
        const content = dropdown.querySelector('.dropdown-content');
        
        if (toggle && content) {
            // Handle mobile dropdown clicks
            toggle.addEventListener('click', function(e) {
                if (window.innerWidth <= 768) {
                    e.preventDefault();
                    content.style.display = content.style.display === 'block' ? 'none' : 'block';
                }
            });
        }
    });
    
    // Active navigation highlighting
    highlightActiveNavigation();
}

// Highlight active navigation based on current page
function highlightActiveNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && currentPath.includes(href.replace(/^\//, ''))) {
            link.classList.add('active');
        }
    });
}

// Smooth scrolling for anchor links
function initSmoothScrolling() {
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                const headerHeight = document.querySelector('.site-header').offsetHeight;
                const targetPosition = targetElement.offsetTop - headerHeight - 20;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Add copy buttons to code blocks
function initCodeCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(codeBlock => {
        const pre = codeBlock.parentElement;
        const button = document.createElement('button');
        
        button.className = 'copy-code-btn';
        button.innerHTML = '<i class="fas fa-copy"></i>';
        button.title = 'Copy code';
        
        button.addEventListener('click', function() {
            const text = codeBlock.textContent;
            
            navigator.clipboard.writeText(text).then(function() {
                button.innerHTML = '<i class="fas fa-check"></i>';
                button.classList.add('copied');
                
                setTimeout(function() {
                    button.innerHTML = '<i class="fas fa-copy"></i>';
                    button.classList.remove('copied');
                }, 2000);
            }).catch(function() {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                
                button.innerHTML = '<i class="fas fa-check"></i>';
                button.classList.add('copied');
                
                setTimeout(function() {
                    button.innerHTML = '<i class="fas fa-copy"></i>';
                    button.classList.remove('copied');
                }, 2000);
            });
        });
        
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
}

// Generate table of contents
function initTableOfContents() {
    const tocContainer = document.querySelector('.toc');
    const headings = document.querySelectorAll('h2, h3, h4, h5, h6');
    
    if (tocContainer && headings.length > 0) {
        const tocList = document.createElement('ul');
        let currentLevel = 2;
        let currentList = tocList;
        const listStack = [tocList];
        
        headings.forEach((heading, index) => {
            const level = parseInt(heading.tagName.substring(1));
            const id = heading.id || `heading-${index}`;
            
            if (!heading.id) {
                heading.id = id;
            }
            
            const listItem = document.createElement('li');
            const link = document.createElement('a');
            link.href = `#${id}`;
            link.textContent = heading.textContent;
            listItem.appendChild(link);
            
            if (level > currentLevel) {
                const newList = document.createElement('ul');
                const lastItem = currentList.lastElementChild;
                if (lastItem) {
                    lastItem.appendChild(newList);
                }
                listStack.push(newList);
                currentList = newList;
            } else if (level < currentLevel) {
                while (listStack.length > 1 && level < currentLevel) {
                    listStack.pop();
                    currentList = listStack[listStack.length - 1];
                    currentLevel--;
                }
            }
            
            currentList.appendChild(listItem);
            currentLevel = level;
        });
        
        const tocTitle = document.createElement('h4');
        tocTitle.textContent = 'Table of Contents';
        tocContainer.appendChild(tocTitle);
        tocContainer.appendChild(tocList);
    }
}

// Simple search functionality
function initSearchFunctionality() {
    // Create search input if it doesn't exist
    const header = document.querySelector('.site-header .header-container');
    const searchContainer = document.createElement('div');
    searchContainer.className = 'search-container';
    searchContainer.innerHTML = `
        <input type="search" id="site-search" placeholder="Search documentation..." aria-label="Search">
        <div class="search-results" id="search-results"></div>
    `;
    
    // Insert search before navigation
    const nav = header.querySelector('.site-nav');
    if (nav && window.innerWidth > 768) {
        header.insertBefore(searchContainer, nav);
    }
    
    const searchInput = document.getElementById('site-search');
    const searchResults = document.getElementById('search-results');
    
    if (searchInput && searchResults) {
        let searchTimeout;
        
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.trim();
            
            if (query.length < 2) {
                searchResults.style.display = 'none';
                return;
            }
            
            searchTimeout = setTimeout(() => {
                performSearch(query, searchResults);
            }, 300);
        });
        
        // Hide search results when clicking outside
        document.addEventListener('click', function(event) {
            if (!searchContainer.contains(event.target)) {
                searchResults.style.display = 'none';
            }
        });
    }
}

// Perform search across page content
function performSearch(query, resultsContainer) {
    const searchableElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li');
    const results = [];
    const queryLower = query.toLowerCase();
    
    searchableElements.forEach(element => {
        const text = element.textContent.toLowerCase();
        if (text.includes(queryLower)) {
            const heading = findNearestHeading(element);
            results.push({
                element: element,
                heading: heading,
                text: element.textContent.substring(0, 100) + '...'
            });
        }
    });
    
    displaySearchResults(results.slice(0, 5), resultsContainer, query);
}

// Find nearest heading for context
function findNearestHeading(element) {
    let current = element;
    while (current && current !== document.body) {
        if (current.tagName && current.tagName.match(/^H[1-6]$/)) {
            return current;
        }
        current = current.previousElementSibling || current.parentElement;
    }
    return null;
}

// Display search results
function displaySearchResults(results, container, query) {
    if (results.length === 0) {
        container.innerHTML = '<div class="search-result">No results found</div>';
    } else {
        container.innerHTML = results.map(result => {
            const headingText = result.heading ? result.heading.textContent : 'Page content';
            const highlightedText = highlightSearchTerm(result.text, query);
            
            return `
                <div class="search-result">
                    <div class="search-result-heading">${headingText}</div>
                    <div class="search-result-text">${highlightedText}</div>
                </div>
            `;
        }).join('');
    }
    
    container.style.display = 'block';
}

// Highlight search terms in results
function highlightSearchTerm(text, term) {
    const regex = new RegExp(`(${term})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
}

// Theme toggle functionality
function initThemeToggle() {
    const themeToggle = document.createElement('button');
    themeToggle.className = 'theme-toggle';
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    themeToggle.title = 'Toggle dark mode';
    
    const header = document.querySelector('.site-header .header-container');
    if (header) {
        header.appendChild(themeToggle);
    }
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeToggleIcon(savedTheme);
    }
    
    themeToggle.addEventListener('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeToggleIcon(newTheme);
    });
}

// Update theme toggle icon
function updateThemeToggleIcon(theme) {
    const themeToggle = document.querySelector('.theme-toggle');
    if (themeToggle) {
        themeToggle.innerHTML = theme === 'dark' ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
    }
}

// Initialize animations and scroll effects
function initAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animateElements = document.querySelectorAll('.feature-card, .alert, .toc');
    animateElements.forEach(element => {
        observer.observe(element);
    });
    
    // Scroll progress indicator
    createScrollProgressIndicator();
    
    // Parallax effect for hero section
    initParallaxEffect();
}

// Create scroll progress indicator
function createScrollProgressIndicator() {
    const progressBar = document.createElement('div');
    progressBar.className = 'scroll-progress';
    document.body.appendChild(progressBar);
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercent = (scrollTop / docHeight) * 100;
        
        progressBar.style.width = scrollPercent + '%';
    });
}

// Parallax effect for hero section
function initParallaxEffect() {
    const hero = document.querySelector('.hero');
    
    if (hero) {
        window.addEventListener('scroll', function() {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            
            hero.style.transform = `translateY(${rate}px)`;
        });
    }
}

// Utility function to debounce events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle window resize events
window.addEventListener('resize', debounce(function() {
    // Recalculate any size-dependent functionality
    const searchContainer = document.querySelector('.search-container');
    const nav = document.querySelector('.site-nav');
    
    if (searchContainer && nav) {
        if (window.innerWidth <= 768) {
            searchContainer.style.display = 'none';
        } else {
            searchContainer.style.display = 'block';
        }
    }
}, 250));

// Add CSS for dynamic elements
const dynamicStyles = `
.copy-code-btn {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 0.5rem;
    cursor: pointer;
    opacity: 0;
    transition: var(--transition);
    color: var(--text-secondary);
}

pre:hover .copy-code-btn {
    opacity: 1;
}

.copy-code-btn:hover {
    background: var(--primary-color);
    color: var(--text-inverse);
}

.copy-code-btn.copied {
    background: var(--success-color);
    color: var(--text-inverse);
}

.search-container {
    position: relative;
    margin-right: 1rem;
}

#site-search {
    padding: 0.5rem 1rem;
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    background: var(--bg-primary);
    color: var(--text-primary);
    width: 250px;
    font-size: 0.875rem;
}

#site-search:focus {
    outline: 2px solid var(--primary-color);
    border-color: var(--primary-color);
}

.search-results {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-lg);
    max-height: 300px;
    overflow-y: auto;
    z-index: 1002;
    display: none;
}

.search-result {
    padding: 0.75rem;
    border-bottom: 1px solid var(--border-color);
    cursor: pointer;
}

.search-result:hover {
    background: var(--bg-secondary);
}

.search-result:last-child {
    border-bottom: none;
}

.search-result-heading {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.875rem;
    margin-bottom: 0.25rem;
}

.search-result-text {
    color: var(--text-secondary);
    font-size: 0.8rem;
}

.search-result mark {
    background: var(--primary-color);
    color: var(--text-inverse);
    padding: 0.1rem 0.2rem;
    border-radius: 2px;
}

.theme-toggle {
    width: 36px;
    height: 36px;
    border: none;
    border-radius: 50%;
    background: var(--bg-secondary);
    color: var(--text-secondary);
    cursor: pointer;
    transition: var(--transition);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: 1rem;
}

.theme-toggle:hover {
    background: var(--primary-color);
    color: var(--text-inverse);
    transform: translateY(-2px);
}

.scroll-progress {
    position: fixed;
    top: 0;
    left: 0;
    width: 0%;
    height: 3px;
    background: var(--primary-color);
    z-index: 1003;
    transition: width 0.1s ease;
}

.animate-in {
    animation: fadeInUp 0.6s ease forwards;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-theme="dark"] {
    --text-primary: #f9fafb;
    --text-secondary: #d1d5db;
    --text-muted: #9ca3af;
    --text-inverse: #111827;
    
    --bg-primary: #111827;
    --bg-secondary: #1f2937;
    --bg-tertiary: #374151;
    --bg-code: #0f172a;
    
    --border-color: #374151;
    --border-dark: #4b5563;
}
`;

// Inject dynamic styles
const styleSheet = document.createElement('style');
styleSheet.textContent = dynamicStyles;
document.head.appendChild(styleSheet); 