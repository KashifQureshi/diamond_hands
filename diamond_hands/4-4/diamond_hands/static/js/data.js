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
	$about.removeClass();
	$analytics.removeClass();
	$data.addClass("active");
}