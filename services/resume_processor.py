import os
import re
from docx import Document
import PyPDF2
import json

class ResumeProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_text(self, file_path):
        """Extract text from resume file"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_pdf_text(file_path)
        elif file_ext == '.docx':
            return self._extract_docx_text(file_path)
        elif file_ext == '.txt':
            return self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _extract_pdf_text(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        return text
    
    def _extract_docx_text(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
        return text
    
    def _extract_txt_text(self, file_path):
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
        return text
    
    def parse_resume(self, text):
        """Parse resume text and extract structured data"""
        resume_data = {
            'name': self._extract_name(text),
            'email': self._extract_email(text),
            'phone': self._extract_phone(text),
            'skills': self._extract_skills(text),
            'experience': self._extract_experience(text),
            'education': self._extract_education(text),
            'summary': self._extract_summary(text),
            'raw_text': text
        }
        return resume_data
    
    def _extract_name(self, text):
        """Extract name from resume text"""
        lines = text.split('\n')
        # Usually the name is in the first few lines
        for line in lines[:5]:
            line = line.strip()
            if line and not any(char in line for char in ['@', ':', '(', ')', '+', '-']) and len(line.split()) <= 4:
                # Check if it looks like a name (contains only letters and spaces)
                if re.match(r'^[A-Za-z\s]+$', line):
                    return line
        return "Name not found"
    
    def _extract_email(self, text):
        """Extract email from resume text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else "Email not found"
    
    def _extract_phone(self, text):
        """Extract phone number from resume text"""
        phone_patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',
            r'\b\(\d{3}\)\s?\d{3}-\d{4}\b',
            r'\b\d{3}\.\d{3}\.\d{4}\b',
            r'\b\+\d{1,3}\s?\d{3}\s?\d{3}\s?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                return phones[0]
        return "Phone not found"
    
    def _extract_skills(self, text):
        """Extract skills from resume text"""
        skills_section = self._find_section(text, ['skills', 'technical skills', 'core competencies'])
        if not skills_section:
            return []
        
        # Common programming languages and technologies
        tech_keywords = [
            'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb',
            'aws', 'docker', 'kubernetes', 'git', 'html', 'css', 'angular',
            'vue.js', 'php', 'ruby', 'go', 'rust', 'c++', 'c#', 'swift',
            'kotlin', 'flutter', 'django', 'flask', 'spring', 'express',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'machine learning', 'data science', 'artificial intelligence'
        ]
        
        found_skills = []
        text_lower = skills_section.lower()
        
        for skill in tech_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return found_skills
    
    def _extract_experience(self, text):
        """Extract work experience from resume text"""
        experience_section = self._find_section(text, ['experience', 'work experience', 'professional experience'])
        if not experience_section:
            return []
        
        # Split by common patterns for job entries
        job_entries = []
        lines = experience_section.split('\n')
        current_job = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains dates (likely a job entry)
            if re.search(r'\d{4}', line):
                if current_job:
                    job_entries.append(current_job)
                current_job = {'raw': line}
            elif current_job:
                current_job['raw'] = current_job.get('raw', '') + ' ' + line
        
        if current_job:
            job_entries.append(current_job)
        
        return job_entries
    
    def _extract_education(self, text):
        """Extract education from resume text"""
        education_section = self._find_section(text, ['education', 'academic background'])
        if not education_section:
            return []
        
        education_entries = []
        lines = education_section.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (any(degree in line.lower() for degree in ['bachelor', 'master', 'phd', 'degree', 'university', 'college'])):
                education_entries.append(line)
        
        return education_entries
    
    def _extract_summary(self, text):
        """Extract summary/objective from resume text"""
        summary_section = self._find_section(text, ['summary', 'objective', 'profile', 'about'])
        if summary_section:
            # Return first paragraph of summary
            paragraphs = summary_section.split('\n\n')
            return paragraphs[0] if paragraphs else summary_section
        return "Summary not found"
    
    def _find_section(self, text, section_keywords):
        """Find a specific section in the resume text"""
        lines = text.split('\n')
        section_start = -1
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in section_keywords):
                section_start = i
                break
        
        if section_start == -1:
            return None
        
        # Find the end of the section (next section or end of text)
        section_end = len(lines)
        for i in range(section_start + 1, len(lines)):
            line = lines[i].strip()
            if line and line.isupper() and len(line) > 3:  # Likely a section header
                section_end = i
                break
        
        section_text = '\n'.join(lines[section_start:section_end])
        return section_text
