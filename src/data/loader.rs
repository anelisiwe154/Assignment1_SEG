use std::fs::File;
use std::io::Read;
use zip::ZipArchive;
use regex::Regex;

pub fn load_docx(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut doc_xml = archive.by_name("word/document.xml")?;
    let mut content = String::new();
    doc_xml.read_to_string(&mut content)?;

    // Remove XML tags
    let re = Regex::new(r"<[^>]+>")?;
    let text = re.replace_all(&content, " ").to_string();

    Ok(text)
}


