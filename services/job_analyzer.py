class JobAnalyzer:
    def __init__(self):
        pass

    def analyze_job(self, job_description):
        """Analyze job description to extract key skills and requirements"""
        lines = job_description.split('\n')
        job_requirements = []

        # Keywords for identifying skills
        keywords = ['skills', 'requirements', 'techniques', 'technologies',
                    'languages', 'tools', 'experience', 'proficiency', 'competencies']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords):
                job_requirements.append(line.strip())
        
        return job_requirements
