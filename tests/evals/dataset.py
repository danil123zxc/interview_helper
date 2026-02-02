"""Dataset examples for subagent evals (no outputs)."""

EXAMPLES = [{'inputs': {'context': {'experience_level': 'mid',
                         'resume': 'Michael Gebauer\n'
                                   "I'm a back-end developer with about 3 years of experience. My core stack "
                                   'includes PHP, Python, Ruby, Java, Rust, C++, and web app testing. At '
                                   'LIMEX Technologies in London (2017-2020), I built and maintained client '
                                   'web applications, gathered requirements directly from customers, '
                                   'troubleshot issues, and ran UI performance tests while collaborating '
                                   'with designers and sys admins. I also trained new team members and was '
                                   'recognized for delivering projects on time and within budget. Key '
                                   'projects included multiple client sites and internal systems where I '
                                   'improved stability through troubleshooting and test coverage. Impact: '
                                   'consistent on-time delivery and measurable performance improvements from '
                                   'UI testing and bug fixes.',
                         'role': 'back-end developer',
                         'years_of_experience': 3},
             'example_id': 'ex-001',
             'user_input': 'Job posting: API/Backend Software Engineer at Particle Health. The role focuses '
                           'on designing, building, and maintaining RESTful APIs and backend services '
                           "primarily in Go, with some Python. You'll implement "
                           'authentication/authorization, data validation, API documentation, caching, and '
                           'performance improvements, while handling sensitive healthcare data with strong '
                           'privacy and security practices. The role collaborates closely with data '
                           'engineering and product teams, supports customer integrations, and contributes '
                           'to architecture and code reviews. Requirements include 3-5 years of backend/API '
                           'experience, strong Go, working Python, SQL/NoSQL, cloud experience '
                           '(AWS/GCP/Azure), Docker/Kubernetes, OAuth/JWT, and solid communication.'}},
 {'inputs': {'context': {'experience_level': 'senior',
                         'resume': 'Thomas Lee Holkain\n'
                                   "I'm a data engineer with roughly 10 years of experience in building data "
                                   'pipelines and data warehouses. My stack centers on SQL, BigQuery, '
                                   'Hadoop, and large-scale data pipeline architecture. At Google '
                                   '(2020-present), I built scalable pipelines that cut processing time by '
                                   'about 65% and helped design a warehouse that accelerated analytics and '
                                   'reporting. At Microsoft (2015-2020), I streamlined data acquisition from '
                                   'multiple sources, improving data accuracy and completeness by about 35%, '
                                   'and resolved data quality issues with cross-functional teams. Earlier, I '
                                   'worked as a data engineering intern focused on analysis and database '
                                   'maintenance. Projects include enterprise pipeline modernization and '
                                   'warehouse redesigns for analytics stakeholders. Impact: significant '
                                   'gains in processing speed and data quality.',
                         'role': 'data engineer',
                         'years_of_experience': 10},
             'example_id': 'ex-002',
             'user_input': 'Job posting: Data Engineer at Arkana Laboratories. You will help build and '
                           'maintain the foundation of a next-generation data platform by designing and '
                           'optimizing data pipelines, data warehouse architecture, and data integration '
                           'processes. The role works with structured and unstructured data sources, '
                           'partners with analytics and backend teams, and supports reporting, analytics, '
                           'and future ML use cases, including a large external dataset integration '
                           'initiative.'}},
 {'inputs': {'context': {'experience_level': 'mid',
                         'resume': 'Victor Frank\n'
                                   "I'm a machine learning engineer with around 5 years of experience. My "
                                   'toolkit includes Python, R, random forests, neural networks, Rasa, '
                                   'Keras, NLTK, and Tesseract OCR. At BOT Shreyasi, I built a '
                                   'conversational bot with Rasa and owned design docs, development, '
                                   'testing, and deployment. As a machine learning intern at Henry Harvin '
                                   'Education (2019), I built models using Decision Trees and Keras and '
                                   'helped lead the ML team from data prep through deployment. Projects '
                                   'include a resume parser using OCR + NLTK and a Python web app (MyBlog) '
                                   'focused on user engagement. Impact: delivered end-to-end ML deployments '
                                   'and automated resume parsing for faster screening.',
                         'role': 'machine learning engineer',
                         'years_of_experience': 5},
             'example_id': 'ex-003',
             'user_input': 'Job posting: Machine Learning Engineer at AI Fund (RealAvatar). You will partner '
                           'with ML and product stakeholders to design a multimodal AI platform for audio, '
                           'video, and text; implement pipeline components; and build low-latency streaming '
                           'Python pipelines. The role includes working with datasets from SQL/Postgres and '
                           'vector stores. Requirements include 5+ years of relevant experience, data '
                           'pipeline engineering, model development and optimization (including LLM '
                           'workflows/agents), production deployment and monitoring, cloud experience '
                           '(GCP/AWS), and a CS/Engineering background.'}},
 {'inputs': {'context': {'experience_level': 'junior',
                         'resume': 'Shay Software\n'
                                   "I'm a software engineer with about 2 years of experience across Java and "
                                   'web stacks. My stack includes Java, Python, Ruby, C/C++, R, J2EE, Rails, '
                                   'HTML/CSS, Oracle/MySQL/Postgres, Spring/Hibernate, REST/SOAP, JUnit, '
                                   'Jenkins, and AWS. At Informatica (2018), I built subscription-based '
                                   'notifications for build errors, created a Jenkins plugin to purge build '
                                   'queues, and shipped a utility that reduced build-server environment '
                                   'issues. At Wipro (2016-2017), I migrated client apps to the cloud, built '
                                   'GWT front-end screens and Java business logic, ran JUnit/JMeter tests, '
                                   'and supported deployments across SIT/DIT/UAT. Projects include a Library '
                                   'Management System (Oracle + Java GUI), a recommendation system in R, '
                                   'Expertiza features in Rails, and a Jenkins-based CI pipeline. Impact: '
                                   'reduced build environment failures and delivered multiple cloud '
                                   'application rewrites with CI improvements.',
                         'role': 'software engineer',
                         'years_of_experience': 2},
             'example_id': 'ex-004',
             'user_input': 'Job posting: Machine Learning Engineer at Filevine. Responsibilities include '
                           'developing and improving NLP models (e.g., summarization, credibility analysis, '
                           'contradiction detection), improving transcription pipelines, automating internal '
                           'processes, and ensuring accuracy, stability, and monitoring around models. '
                           "You'll communicate ML results in a business context and collaborate with "
                           'full-stack teams to ship models to production. Qualifications include 3+ years '
                           'of ML experience with end-to-end ownership, strong NLP and PyTorch, deep '
                           'learning experience, solid software engineering, and comfort with production '
                           'stacks that use Python, React/TypeScript, Kubernetes, and AWS.'}},
 {'inputs': {'context': {'experience_level': 'mid',
                         'resume': 'Hans Benny Bear\n'
                                   "I'm a front-end developer with about 3 years of experience building web "
                                   'UIs. My stack centers on JavaScript (Backbone, jQuery, Underscore), '
                                   'HTML, and CSS. At BudgetMatador (2015-present), I built the product '
                                   'website and UI components, shipping production UI for a startup. I '
                                   'previously worked as a customer-service consultant, which strengthened '
                                   'my user empathy. Projects include a responsive startup website with '
                                   'interactive components. Impact: delivered the production front-end and '
                                   'improved usability through solid client-side architecture.',
                         'role': 'front-end developer',
                         'years_of_experience': 3},
             'example_id': 'ex-005',
             'user_input': 'Job posting: Frontend Engineer at Logz.io. You will turn product ideas into rich '
                           'UI/UX features, write clean and maintainable front-end code, and build web '
                           'applications using React and Node.js. The role collaborates closely with product '
                           'and design and includes performance tuning and quality practices. Requirements '
                           'include 3+ years building web apps, strong JavaScript/TypeScript, React '
                           'expertise, solid HTML/CSS (including SASS or Styled Components), and a focus on '
                           'clean architecture and code quality.'}},
 {'inputs': {'context': {'experience_level': 'mid',
                         'resume': 'Olivier Baudet\n'
                                   "I'm a web developer with around 4 years of experience. My stack includes "
                                   'PHP, MySQL, HTML/CSS/JavaScript, Java, C++, and Adobe Dreamweaver. At '
                                   'IDOX Software in Paris (2017-2019), I led development of client '
                                   'websites, advised customers, supervised a team of seven freelancers, '
                                   'collaborated with PMs and creatives, and presented market research to '
                                   'leadership. At Stewart Travel in London (2014-2017), I designed and '
                                   'implemented new websites end-to-end, maintained existing sites, trained '
                                   'team members, and earned Employee of the Month for quality delivery. '
                                   'Projects include client website builds and redesigns across travel and '
                                   'software domains. Impact: led multi-site launches and improved delivery '
                                   'quality while mentoring a small dev team.',
                         'role': 'web developer',
                         'years_of_experience': 4},
             'example_id': 'ex-006',
             'user_input': 'Job posting: Senior Frontend Software Engineer at Highspot. You will own the '
                           'full lifecycle of customer-facing features, drive frontend infrastructure '
                           'improvements, collaborate with product/design/backend, and build cross-browser '
                           'interactive web apps with JavaScript, HTML, and CSS while consuming REST APIs. '
                           'The role includes mentoring and technical leadership. Requirements include 6+ '
                           'years of software development, 3+ years with React, experience architecting '
                           'mid-to-large web applications, and strong communication.'}},
 {'inputs': {'context': {'experience_level': 'mid',
                         'resume': 'Parat Gason\n'
                                   "I'm a site reliability engineer with about 4 years of experience in "
                                   'automation, monitoring, and reliability. My background includes '
                                   'microservices, pipeline frameworks, monitoring/testing, automation '
                                   'tooling, and web validation. At Worldpay (2017-2019), I built '
                                   'microservices and pipeline frameworks, provided system support and '
                                   'troubleshooting, created automation tools, and managed live-service '
                                   'monitoring. At Starling Bank (2015-2017), I implemented '
                                   'monitoring/testing, ran security assessments for web and mobile '
                                   'services, automated processes, and managed incident resolution and error '
                                   'budgets. Projects centered on monitoring and automation initiatives to '
                                   'improve reliability. Impact: improved service reliability through '
                                   'automation and faster troubleshooting.',
                         'role': 'site reliability engineer',
                         'years_of_experience': 4},
             'example_id': 'ex-007',
             'user_input': 'Job posting: Site Reliability Engineer at Breeze Airways. You will maintain '
                           'availability and reliability of critical platform services, manage production '
                           'systems and incidents, automate operations, build monitoring and '
                           'infrastructure-as-code, and deploy/maintain CI/CD pipelines for cloud-native '
                           'microservices. Requirements include 4+ years running highly available systems, '
                           'scripting for automation, containerization, AWS experience, Git, CI/CD, '
                           'configuration management, and troubleshooting distributed systems.'}},
 {'inputs': {'context': {'experience_level': 'senior',
                         'resume': 'Kael Lightningstrike\n'
                                   "I'm a C#/.NET software engineer with roughly 8 years of experience. My "
                                   'stack includes C#, .NET Framework, ASP.NET, Visual Studio, and Git. At '
                                   'Amazon (2021-present), I built scalable web apps, improved application '
                                   'performance by about 20%, collaborated on requirements and design, '
                                   'resolved complex issues, and conducted code reviews. Earlier at Amazon '
                                   '(2016-2021), I maintained C# applications, implemented OOP-based '
                                   'solutions, tested/debugged code, and translated user requirements into '
                                   'specs. Projects include web application modernization and performance '
                                   'optimization. Impact: measurable performance gains and stronger code '
                                   'quality through reviews and collaboration.',
                         'role': 'software engineer',
                         'years_of_experience': 8},
             'example_id': 'ex-008',
             'user_input': 'Job posting: Staff Platform Engineer at Aktos. You will maintain AWS '
                           'infrastructure, build scalable backend features in Python/Django, implement API '
                           'and SFTP integrations, optimize Postgres performance, manage CI pipelines and '
                           'alerting, and participate in on-call. Requirements include 5+ years of software '
                           'development, 2+ years of SRE/DevOps, Kubernetes maintained via IaC (Terraform '
                           'preferred), 2+ years of Python, strong RDBMS skills (Postgres), and deep AWS '
                           'knowledge.'}},
 {'inputs': {'context': {'experience_level': 'mid',
                         'resume': 'Marco Muller\n'
                                   "I'm a product manager with about 5 years of experience and a strong "
                                   'analytics toolkit (GoodData, Google Analytics, Aha!, Segment, MS '
                                   'Office). At FinancialForce (2016-2019), I led market research and '
                                   'competitive analysis, managed a 20-person sales team, partnered with '
                                   'marketing on campaigns, and ensured policy compliance, driving revenue '
                                   'growth of 20% in 2017 and 30% in 2018. At First Utility (2014-2016), I '
                                   'identified market and customer needs, collaborated on catalogs and '
                                   'digital content, updated the website, supported weekly planning, and '
                                   'earned Employee of the Month twice. Earlier, I interned at GE building '
                                   'promotional campaigns and market research reports. Projects include '
                                   'go-to-market materials and analytics dashboards for sales performance. '
                                   'Impact: strong revenue growth and improved sales-team effectiveness.',
                         'role': 'product manager',
                         'years_of_experience': 5},
             'example_id': 'ex-009',
             'user_input': 'Job posting: Product Manager at Triple Whale. You will lead product development '
                           'from concept to launch, collaborate with engineering/design/marketing, define '
                           'requirements and detailed specs, analyze market and competition, prioritize '
                           'features, and conduct user research to inform decisions. Requirements include 3+ '
                           'years of PM experience in tech, agile/product lifecycle knowledge, proficiency '
                           'with tools like Jira/Confluence, strong communication and analytical skills, and '
                           'a CS/Engineering or related degree.'}},
 {'inputs': {'context': {'experience_level': 'mid',
                         'resume': 'Wendy Williams\n'
                                   "I'm a software engineer with around 6 years of experience across web, "
                                   'firmware, and data systems. My stack includes web development, firmware, '
                                   'server development, MapReduce, monitoring/alerting, and data pipelines. '
                                   'At Lady Technologies (2018-present), I worked across web, firmware, '
                                   'test, and server development for an IoT device. At Kickresume '
                                   '(2017-2018), I built web applications for resumes, cover letters, and '
                                   'personal websites. I also completed Google internships where I rewrote a '
                                   'pipeline using MapReduce and set up monitoring/alerting for a data '
                                   'pipeline, and earlier developed meteorological models and visualizations '
                                   'at MicroStep-MIS. Projects include IoT device software and data pipeline '
                                   'monitoring tools. Impact: improved pipeline reliability and delivered '
                                   'end-to-end IoT software features.',
                         'role': 'software engineer',
                         'years_of_experience': 6},
             'example_id': 'ex-010',
             'user_input': 'Job posting: Senior Technical Product Manager, Networking at Ditto. You will '
                           'drive the roadmap for networking and transport capabilities, balance '
                           'performance/reliability/complexity tradeoffs, coordinate across PM and '
                           'engineering teams, engage customers on networking needs, and translate '
                           'requirements into technical specs. Requirements include a CS/Engineering degree '
                           '(or equivalent experience) and exposure to networking or distributed systems '
                           'concepts.'}}]
