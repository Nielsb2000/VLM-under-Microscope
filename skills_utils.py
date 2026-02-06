"""Utilities for discovering, parsing, and validating skills."""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import yaml


class SkillMetadata:
    """Represents skill metadata parsed from frontmatter."""
    
    def __init__(self, name: str, description: str, path: Path):
        self.name = name
        self.description = description
        self.path = path
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path)
        }


def load_skills_files() -> Dict[str, Dict]:
    """Load all SKILL.md files from the skills directory as FileData for the agent.
    
    Returns:
        Dictionary mapping virtual paths to FileData dicts with content (list[str]), created_at, and modified_at
    """
    skills_files = {}
    skills_path = Path("./skills")
    
    if not skills_path.exists():
        return skills_files
    
    # Use UTC timezone for consistency
    now = datetime.now(timezone.utc).isoformat()
    
    for skill_folder in sorted(skills_path.iterdir()):
        if not skill_folder.is_dir():
            continue
        
        skill_file = skill_folder / "SKILL.md"
        if skill_file.exists():
            with open(skill_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            virtual_path = f"/skills/{skill_folder.name}/SKILL.md"
            # Format content as list of lines for StateBackend FileData structure
            skills_files[virtual_path] = {
                "content": content.split("\n"),
                "created_at": now,
                "modified_at": now,
            }
    
    return skills_files


def parse_frontmatter(content: str) -> Optional[Dict]:
    """
    Extract YAML frontmatter from markdown content.
    
    Args:
        content: Full markdown content with frontmatter
    
    Returns:
        Dictionary of parsed YAML, or None if no valid frontmatter
    """
    # Match YAML frontmatter between --- markers
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    
    if not match:
        return None
    
    frontmatter_text = match.group(1)
    
    try:
        return yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError:
        return None


def read_skill_metadata(skill_path: Path) -> Optional[SkillMetadata]:
    """
    Read and parse skill metadata from a skill directory.
    
    Args:
        skill_path: Path to skill directory containing SKILL.md
    
    Returns:
        SkillMetadata object, or None if invalid
    """
    skill_file = skill_path / "SKILL.md"
    
    if not skill_file.exists():
        return None
    
    try:
        content = skill_file.read_text(encoding='utf-8')
        frontmatter = parse_frontmatter(content)
        
        if not frontmatter:
            return None
        
        name = frontmatter.get('name')
        description = frontmatter.get('description')
        
        if not name or not description:
            return None
        
        return SkillMetadata(name=name, description=description, path=skill_path)
    
    except Exception:
        return None


def discover_skills(skills_dir: Path) -> List[SkillMetadata]:
    """
    Discover all skills in a directory.
    
    Args:
        skills_dir: Path to directory containing skill folders
    
    Returns:
        List of SkillMetadata objects
    """
    skills = []
    
    if not skills_dir.exists():
        return skills
    
    for skill_folder in sorted(skills_dir.iterdir()):
        if not skill_folder.is_dir():
            continue
        
        metadata = read_skill_metadata(skill_folder)
        if metadata:
            skills.append(metadata)
    
    return skills


def validate_skill(skill_path: Path) -> List[str]:
    """
    Validate a skill directory.
    
    Args:
        skill_path: Path to skill directory
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    skill_file = skill_path / "SKILL.md"
    
    # Check if SKILL.md exists
    if not skill_file.exists():
        errors.append(f"SKILL.md not found in {skill_path}")
        return errors
    
    # Check if file is readable
    try:
        content = skill_file.read_text(encoding='utf-8')
    except Exception as e:
        errors.append(f"Cannot read SKILL.md: {e}")
        return errors
    
    # Check frontmatter exists
    frontmatter = parse_frontmatter(content)
    if frontmatter is None:
        errors.append("No valid YAML frontmatter found (must be between --- markers)")
        return errors
    
    # Check required fields
    if not frontmatter.get('name'):
        errors.append("Missing required 'name' field in frontmatter")
    
    if not frontmatter.get('description'):
        errors.append("Missing required 'description' field in frontmatter")
    
    # Check content exists after frontmatter
    match = re.match(r'^---\s*\n.*?\n---\s*\n(.*)', content, re.DOTALL)
    if not match or not match.group(1).strip():
        errors.append("No skill content found after frontmatter")
    
    return errors


def generate_available_skills_xml(skills: List[SkillMetadata]) -> str:
    """
    Generate <available_skills> XML block for agent prompt.
    
    Args:
        skills: List of SkillMetadata objects
    
    Returns:
        XML string representing available skills
    """
    if not skills:
        return "<available_skills></available_skills>"
    
    xml_lines = ["<available_skills>"]
    
    for skill in skills:
        xml_lines.append("  <skill>")
        xml_lines.append(f"    <name>{skill.name}</name>")
        xml_lines.append(f"    <description>{skill.description}</description>")
        xml_lines.append(f"    <location>{skill.path / 'SKILL.md'}</location>")
        xml_lines.append("  </skill>")
    
    xml_lines.append("</available_skills>")
    
    return "\n".join(xml_lines)


def get_skills_summary(skills: List[SkillMetadata]) -> str:
    """
    Generate a human-readable summary of available skills.
    
    Args:
        skills: List of SkillMetadata objects
    
    Returns:
        Formatted string with skill names and descriptions
    """
    if not skills:
        return "No skills available"
    
    lines = [f"Available skills ({len(skills)}):"]
    for skill in skills:
        lines.append(f"  • {skill.name}: {skill.description}")
    
    return "\n".join(lines)
