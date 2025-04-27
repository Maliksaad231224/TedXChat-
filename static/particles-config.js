particlesJS("particles-js", {
    particles: {
        number: { value: 120, density: { enable: true, value_area: 900 } },
        color: { value: ["#ff0080", "#00ffff", "#ffcc00"] },
        shape: { type: "circle" },
        opacity: { value: 0.2, random: true },
        size: { value: 8, random: true },
        move: { enable: true, speed: 3, direction: "none", random: false }
    },
    interactivity: {
        events: { onhover: { enable: true, mode: "repulse" } },
        modes: { repulse: { distance: 100, duration: 0.4 } }
    },
    retina_detect: true
});
