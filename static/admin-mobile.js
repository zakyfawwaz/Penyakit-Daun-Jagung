// Admin Mobile Sidebar Toggle Script
// Include this script in admin templates that use the sidebar

document.addEventListener('DOMContentLoaded', function() {
    const mobileSidebarToggle = document.getElementById('mobileSidebarToggle');
    const adminSidebar = document.querySelector('.admin-sidebar');
    
    if (mobileSidebarToggle && adminSidebar) {
        mobileSidebarToggle.addEventListener('click', function(e) {
            e.stopPropagation();
            adminSidebar.classList.toggle('active');
            mobileSidebarToggle.textContent = adminSidebar.classList.contains('active') ? '✕' : '☰';
        });

        // Close sidebar when clicking on a menu item
        const menuItems = adminSidebar.querySelectorAll('.menu-item');
        menuItems.forEach(function(item) {
            item.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    adminSidebar.classList.remove('active');
                    if (mobileSidebarToggle) {
                        mobileSidebarToggle.textContent = '☰';
                    }
                }
            });
        });

        // Close sidebar when clicking outside
        document.addEventListener('click', function(e) {
            if (window.innerWidth <= 768) {
                if (adminSidebar && adminSidebar.classList.contains('active')) {
                    if (!adminSidebar.contains(e.target) && !mobileSidebarToggle.contains(e.target)) {
                        adminSidebar.classList.remove('active');
                        mobileSidebarToggle.textContent = '☰';
                    }
                }
            }
        });

        // Handle window resize
        window.addEventListener('resize', function() {
            if (window.innerWidth > 768) {
                adminSidebar.classList.remove('active');
                if (mobileSidebarToggle) {
                    mobileSidebarToggle.textContent = '☰';
                }
            }
        });
    }
});

