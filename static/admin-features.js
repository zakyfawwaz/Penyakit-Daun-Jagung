// Admin Dashboard Features: Search, Language, Theme Toggle, Notifications

document.addEventListener('DOMContentLoaded', function() {
    // ==================== SEARCH FUNCTIONALITY ====================
    const searchInput = document.querySelector('.search-bar input');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase().trim();
            const tables = document.querySelectorAll('.table tbody');
            
            tables.forEach(table => {
                const rows = table.querySelectorAll('tr');
                let hasVisibleRows = false;
                
                rows.forEach(row => {
                    const text = row.textContent.toLowerCase();
                    if (text.includes(searchTerm)) {
                        row.style.display = '';
                        hasVisibleRows = true;
                    } else {
                        row.style.display = 'none';
                    }
                });
                
                // Show "No results" message if needed
                let noResultsRow = table.querySelector('.no-results-row');
                if (!hasVisibleRows && searchTerm !== '') {
                    if (!noResultsRow) {
                        noResultsRow = document.createElement('tr');
                        noResultsRow.className = 'no-results-row';
                        noResultsRow.innerHTML = '<td colspan="100" style="text-align: center; color: #94a3b8; padding: 20px;">No results found</td>';
                        table.appendChild(noResultsRow);
                    }
                    noResultsRow.style.display = '';
                } else if (noResultsRow) {
                    noResultsRow.style.display = 'none';
                }
            });
        });
    }

    // ==================== THEME TOGGLE ====================
    const themeToggle = document.getElementById('themeToggle') || document.querySelector('.header-right button.header-action:nth-of-type(2)');
    if (themeToggle) {
        // Load saved theme preference
        const savedTheme = localStorage.getItem('adminTheme') || 'light';
        if (savedTheme === 'dark') {
            document.body.classList.add('admin-dark-theme');
            themeToggle.textContent = '🌙';
        } else {
            themeToggle.textContent = '☀️';
        }

        // Handle theme toggle
        themeToggle.addEventListener('click', function() {
            document.body.classList.toggle('admin-dark-theme');
            const isDark = document.body.classList.contains('admin-dark-theme');
            localStorage.setItem('adminTheme', isDark ? 'dark' : 'light');
            themeToggle.textContent = isDark ? '🌙' : '☀️';
        });
    }

    // ==================== NOTIFICATION SYSTEM ====================
    const notificationBtn = document.getElementById('notificationBtn') || document.querySelector('.header-right button.header-action[aria-label="Notifications"]') || document.querySelector('.header-right button.header-action:nth-of-type(2)');
    if (notificationBtn) {
        // Create notification dropdown
        let notificationDropdown = document.querySelector('.notification-dropdown');
        
        // Update dropdown position function
        const updateDropdownPosition = () => {
            if (notificationDropdown && notificationBtn) {
                const rect = notificationBtn.getBoundingClientRect();
                notificationDropdown.style.top = (rect.bottom + 10) + 'px';
                notificationDropdown.style.right = (window.innerWidth - rect.right) + 'px';
            }
        };
        
        // Initialize notification count
        let notificationCount = 2; // Number of unread notifications
        
        if (!notificationDropdown) {
            notificationDropdown = document.createElement('div');
            notificationDropdown.className = 'notification-dropdown';
            notificationDropdown.innerHTML = `
                <div class="notification-header">
                    <h4>Notifications</h4>
                    <button class="notification-mark-all" type="button">Mark all as read</button>
                </div>
                <div class="notification-list">
                    <div class="notification-item unread" data-id="1">
                        <div class="notification-icon">🔔</div>
                        <div class="notification-content">
                            <div class="notification-title">New Detection</div>
                            <div class="notification-text">A new corn disease detection has been submitted</div>
                            <div class="notification-time">2 minutes ago</div>
                        </div>
                    </div>
                    <div class="notification-item unread" data-id="2">
                        <div class="notification-icon">👤</div>
                        <div class="notification-content">
                            <div class="notification-title">New User Registered</div>
                            <div class="notification-text">A new user has joined the platform</div>
                            <div class="notification-time">1 hour ago</div>
                        </div>
                    </div>
                    <div class="notification-item" data-id="3">
                        <div class="notification-icon">📊</div>
                        <div class="notification-content">
                            <div class="notification-title">Daily Report</div>
                            <div class="notification-text">Daily statistics report is ready</div>
                            <div class="notification-time">3 hours ago</div>
                        </div>
                    </div>
                </div>
                <div class="notification-footer">
                    <a href="#" class="notification-view-all">View all notifications</a>
                </div>
            `;
            document.body.appendChild(notificationDropdown);
            
            // Update position on scroll and resize
            window.addEventListener('scroll', updateDropdownPosition, true);
            window.addEventListener('resize', updateDropdownPosition);
        }

        // Function to update notification badge
        function updateNotificationBadge(count) {
            // Remove existing badge
            const existingBadge = notificationBtn.querySelector('.notification-badge');
            if (existingBadge) {
                existingBadge.remove();
            }

            // Add badge if there are unread notifications
            if (count > 0) {
                const badge = document.createElement('span');
                badge.className = 'notification-badge';
                badge.textContent = count > 9 ? '9+' : count;
                notificationBtn.appendChild(badge);
            }
        }

        // Initialize badge
        updateNotificationBadge(notificationCount);

        // Toggle notification dropdown
        notificationBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            e.preventDefault();
            updateDropdownPosition();
            notificationDropdown.classList.toggle('active');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!notificationBtn.contains(e.target) && !notificationDropdown.contains(e.target)) {
                notificationDropdown.classList.remove('active');
            }
        });

        // Mark all as read
        const markAllBtn = notificationDropdown.querySelector('.notification-mark-all');
        if (markAllBtn) {
            markAllBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                e.preventDefault();
                const unreadItems = notificationDropdown.querySelectorAll('.notification-item.unread');
                unreadItems.forEach(item => {
                    item.classList.remove('unread');
                });
                notificationCount = 0;
                updateNotificationBadge(notificationCount);
            });
        }

        // Mark individual notification as read
        const notificationItems = notificationDropdown.querySelectorAll('.notification-item');
        notificationItems.forEach(item => {
            item.addEventListener('click', function(e) {
                e.stopPropagation();
                if (this.classList.contains('unread')) {
                    this.classList.remove('unread');
                    notificationCount = Math.max(0, notificationCount - 1);
                    updateNotificationBadge(notificationCount);
                }
            });
        });

        // View all notifications link
        const viewAllLink = notificationDropdown.querySelector('.notification-view-all');
        if (viewAllLink) {
            viewAllLink.addEventListener('click', function(e) {
                e.preventDefault();
                // You can redirect to a notifications page here
                console.log('View all notifications clicked');
                notificationDropdown.classList.remove('active');
            });
        }
    }
});

