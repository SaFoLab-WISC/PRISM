window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

	// Navbar burger toggle
	$('.navbar-burger').on('click', function() {
		var target = $(this).data('target');
		$(this).toggleClass('is-active');
		$('#' + target).toggleClass('is-active');
	});

	// Active link highlighting on scroll
	const sections = ['overview','method','results','examples','paper','citation'];
	const navLinks = sections.map(id => ({ id, el: $("a.navbar-item[href='#"+id+"']") }));

	function setActive() {
		let scrollPos = $(window).scrollTop();
		let chosen = null;
		sections.forEach(id => {
			const top = $('#'+id).offset() ? $('#'+id).offset().top - 120 : Infinity;
			if(scrollPos >= top) chosen = id;
		});
		navLinks.forEach(l => l.el.toggleClass('active', l.id === chosen));
	}
	$(window).on('scroll', setActive);
	setActive();

	// Copy citation
	$('#copy-citation').on('click', function(){
		const bib = $('#bibtex-entry').text();
		navigator.clipboard.writeText(bib).then(()=>{
			const btn = $(this);
			const original = btn.text();
			btn.addClass('is-success').text('Copied!');
			setTimeout(()=>{ btn.removeClass('is-success').text(original); }, 1800);
		});
	});

	// Back to top button logic
	const backBtn = $('#back-to-top');
	function toggleBackBtn(){
		if($(window).scrollTop() > 400){ backBtn.addClass('is-visible').show(); }
		else { backBtn.removeClass('is-visible'); setTimeout(()=>{ if(!backBtn.hasClass('is-visible')) backBtn.hide(); }, 250);} }
	$(window).on('scroll', toggleBackBtn);
	toggleBackBtn();
	backBtn.on('click', function(){ window.scrollTo({top:0, behavior:'smooth'}); });
})
