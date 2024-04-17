document.querySelector('.menu-button').addEventListener('click', function() {
    var dropdownMenu = document.querySelector('.dropdown-content');
    if (dropdownMenu.classList.contains('show')) {
        dropdownMenu.classList.remove('show');
    } else {
        dropdownMenu.classList.add('show');
    }
});
