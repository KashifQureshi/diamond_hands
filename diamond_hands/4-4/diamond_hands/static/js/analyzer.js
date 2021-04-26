$(document).ready(function()
{
    start();
});

function start() {
	var $home = $(' #home ');
	var $data = $(' #data ');
	var $analytics = $(' #analytics ');
	var $about = $(' #about ' );
	$home.removeClass();
	$data.removeClass();
	$about.removeClass()
	$analytics.addClass("active");
}