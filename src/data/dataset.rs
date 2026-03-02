use burn::data::dataset::Dataset;
use regex::Regex;

#[derive(Clone, Debug)]
pub struct QaSample {
    pub context: String,
    pub question: String,
    pub answer: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug)]
pub struct QaDataset {
    pub samples: Vec<QaSample>,
}

impl Dataset<QaSample> for QaDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<QaSample> {
        self.samples.get(index).cloned()
    }
}

impl QaDataset {
    /// Convenience: detect year from text (first match of 2024/2025/2026)
    pub fn from_text(doc_text: String) -> Self {
        let year = detect_year(&doc_text).unwrap_or_else(|| "2024".to_string());
        Self::from_text_with_year(doc_text, &year)
    }

    /// Main builder: parse the calendar using the provided year.
    pub fn from_text_with_year(doc_text: String, year: &str) -> Self {
        let normalized = doc_text
            .replace("\r\n", "\n")
            .replace('\t', " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        // match: "JANUARY 2026" etc
        let month_re = Regex::new(&format!(
            r"(?i)\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+{}\b",
            regex::escape(year)
        ))
        .unwrap();

        let day_re = Regex::new(r"\b([1-9]|[12]\d|3[01])\b").unwrap();

        let weekdays = [
            "SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY",
        ];

        let mut month_spans: Vec<(usize, usize, String)> = Vec::new();
        for cap in month_re.captures_iter(&normalized) {
            let m = cap.get(1).unwrap().as_str().to_uppercase();
            let whole = cap.get(0).unwrap();
            month_spans.push((whole.start(), whole.end(), m));
        }

        // If no month headings found, fallback
        if month_spans.is_empty() {
            return Self {
                samples: vec![QaSample {
                    context: doc_text,
                    question: "What is this document about?".to_string(),
                    answer: format!("Calendar events in {}", year),
                    start: 0,
                    end: 0,
                }],
            };
        }

        // Build month sections as slices after each MONTH YEAR header
        let mut month_sections: Vec<(usize, usize, String)> = Vec::new();
        for idx in 0..month_spans.len() {
            let (_start, end, month) = month_spans[idx].clone();
            let next_start = if idx + 1 < month_spans.len() {
                month_spans[idx + 1].0
            } else {
                normalized.len()
            };
            month_sections.push((end, next_start, month));
        }

        let mut events: Vec<(String, String)> = Vec::new();

        for (sec_start, sec_end, month) in month_sections {
            let section = &normalized[sec_start..sec_end];
            let day_matches: Vec<_> = day_re.find_iter(section).collect();

            for k in 0..day_matches.len() {
                let day_m = day_matches[k];
                let day = day_m.as_str();

                let chunk_start = day_m.end();
                let chunk_end = if k + 1 < day_matches.len() {
                    day_matches[k + 1].start()
                } else {
                    section.len()
                };

                let mut chunk = section[chunk_start..chunk_end].trim().to_string();
                if chunk.is_empty() {
                    continue;
                }

                let first_word = chunk
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_uppercase();

                if weekdays.contains(&first_word.as_str()) {
                    continue;
                }

                if chunk.len() > 220 {
                    chunk = chunk.chars().take(220).collect();
                }

                let date_str = format!("{} {} {}", day, month, year);
                events.push((date_str, chunk));
            }
        }

        let mut samples = Vec::new();
        for (date, event) in events.iter() {
            samples.push(QaSample {
                context: doc_text.clone(),
                question: format!("What event happens on {}?", date),
                answer: event.clone(),
                start: 0,
                end: 0,
            });

            samples.push(QaSample {
                context: doc_text.clone(),
                question: format!("When is {}?", event),
                answer: date.clone(),
                start: 0,
                end: 0,
            });
        }

        println!("Extracted {} events", events.len());
        println!("Generated {} training samples", samples.len());

        if samples.is_empty() {
            samples.push(QaSample {
                context: doc_text,
                question: "What is this calendar about?".to_string(),
                answer: format!("Calendar events in {}", year),
                start: 0,
                end: 0,
            });
        }

        Self { samples }
    }
}

fn detect_year(text: &str) -> Option<String> {
    for y in ["2024", "2025", "2026"] {
        if text.contains(y) {
            return Some(y.to_string());
        }
    }
    None
}














