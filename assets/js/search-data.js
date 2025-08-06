// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-projects",
          title: "projects",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "projects-madbus",
          title: 'MADBUS',
          description: "Work completed as part of the Robotics and Automation Summer School at Los Alamos National Laboratory. Worked on and written with Olyvia Hanken-Arlen, Anunth Ramaswami, Jessica Mendez, Colin Sanders, Matthew Hammond, and Dr. Beth Boadman. Supported by Los Alamos National Laboratory and approved for release under LA-UR-24-33374.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/RASS_project/";
            },},{id: "projects-rl-for-high-performance-jumping",
          title: 'RL for High-Performance Jumping',
          description: "Used a curriculum learning framework to teach simulation quadrupeds to jump. Worked on and written with Aryan Naveen and Pranay Varada. Completed for CS1840 - Introduction to Reinforcement Learning",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cs1840_project/";
            },},{id: "projects-semantic-segmentation-of-aerial-photographs",
          title: 'Semantic Segmentation of Aerial Photographs',
          description: "Per-pixel land use classification of sattelite imagery of Mumbai. Completed for CS2831 - Advanced Computer Vision",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cs2831_project/";
            },},{id: "projects-feeder-schools-the-harvard-crimson",
          title: 'Feeder Schools, The Harvard Crimson',
          description: "Written with Elyse Goncalves and Matan Josephy with additional work by Alma T. Barak, Ben Ali H. Brown, Angela S. Chen, Darcy G Lin Elise A. Spenner, Dennis S. Eum, and Neil H. Shah",
          section: "Projects",handler: () => {
              window.location.href = "/projects/feeder_schools/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%67%72%61%79%73%6F%6E%6D%61%72%74%69%6E%30%34@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/GraysonMMartin", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/grayson-martin-4458a4203", "_blank");
        },
      },{
        id: 'social-spotify',
        title: 'Spotify',
        section: 'Socials',
        handler: () => {
          window.open("https://open.spotify.com/user/graysonmartin04", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
