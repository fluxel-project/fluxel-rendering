//! Private WGSL top-level declaration scanning for Material validation.

use super::MaterialError;

pub(super) fn wgsl_top_level_names(source: &str) -> Result<Vec<String>, MaterialError> {
    let mut names = Vec::new();
    let mut position = 0;
    let mut braces = 0_u32;
    while position < source.len() {
        position = skip_trivia(source, position)?;
        let Some(character) = next_char(source, position) else {
            break;
        };
        match character {
            '{' => {
                braces += 1;
                position += 1;
            }
            '}' => {
                braces = braces
                    .checked_sub(1)
                    .ok_or(MaterialError::TopLevelWgslUnparseable)?;
                position += 1;
            }
            '"' => position = skip_string(source, position)?,
            _ if braces == 0 && is_identifier_start(character) => {
                let (keyword, after_keyword) = read_identifier(source, position);
                position = after_keyword;
                let declaration = match keyword {
                    "fn" | "struct" | "const" | "override" | "alias" | "let" => {
                        next_identifier(source, position)?
                    }
                    "var" => next_variable_name(source, position)?,
                    _ => None,
                };
                if let Some((name, after_name)) = declaration {
                    names.push(name.to_owned());
                    position = after_name;
                }
            }
            _ => position += character.len_utf8(),
        }
    }
    (braces == 0)
        .then_some(names)
        .ok_or(MaterialError::TopLevelWgslUnparseable)
}

fn next_variable_name(
    source: &str,
    mut position: usize,
) -> Result<Option<(&str, usize)>, MaterialError> {
    position = skip_trivia(source, position)?;
    if source.as_bytes().get(position) == Some(&b'<') {
        let mut angles = 0_u32;
        loop {
            position = skip_trivia(source, position)?;
            let character =
                next_char(source, position).ok_or(MaterialError::TopLevelWgslUnparseable)?;
            match character {
                '<' => angles += 1,
                '>' => {
                    angles = angles
                        .checked_sub(1)
                        .ok_or(MaterialError::TopLevelWgslUnparseable)?;
                    position += 1;
                    if angles == 0 {
                        break;
                    }
                    continue;
                }
                _ => {}
            }
            position += character.len_utf8();
        }
        if angles != 0 {
            return Err(MaterialError::TopLevelWgslUnparseable);
        }
    }
    next_identifier(source, position)
}

fn next_identifier(source: &str, position: usize) -> Result<Option<(&str, usize)>, MaterialError> {
    let position = skip_trivia(source, position)?;
    Ok(next_char(source, position)
        .filter(|character| is_identifier_start(*character))
        .map(|_| read_identifier(source, position)))
}

fn read_identifier(source: &str, position: usize) -> (&str, usize) {
    let mut end = position;
    for character in source[position..].chars() {
        if end != position && !unicode_ident::is_xid_continue(character) {
            break;
        }
        end += character.len_utf8();
    }
    (&source[position..end], end)
}

fn skip_trivia(source: &str, mut position: usize) -> Result<usize, MaterialError> {
    loop {
        while next_char(source, position).is_some_and(is_whitespace) {
            position += next_char(source, position)
                .expect("checked character")
                .len_utf8();
        }
        if source.as_bytes().get(position..position + 2) == Some(b"//") {
            position += 2;
            while next_char(source, position).is_some_and(|character| character != '\n') {
                position += next_char(source, position)
                    .expect("checked character")
                    .len_utf8();
            }
            continue;
        }
        if source.as_bytes().get(position..position + 2) == Some(b"/*") {
            position += 2;
            let mut comments = 1_u32;
            while comments != 0 && position < source.len() {
                match source.as_bytes().get(position..position + 2) {
                    Some(b"/*") => {
                        comments += 1;
                        position += 2;
                    }
                    Some(b"*/") => {
                        comments -= 1;
                        position += 2;
                    }
                    _ => {
                        position += next_char(source, position)
                            .ok_or(MaterialError::TopLevelWgslUnparseable)?
                            .len_utf8()
                    }
                }
            }
            if comments != 0 {
                return Err(MaterialError::TopLevelWgslUnparseable);
            }
            continue;
        }
        return Ok(position);
    }
}

fn skip_string(source: &str, mut position: usize) -> Result<usize, MaterialError> {
    position += 1;
    while let Some(character) = next_char(source, position) {
        position += character.len_utf8();
        match character {
            '"' => return Ok(position),
            '\\' => {
                position += next_char(source, position)
                    .ok_or(MaterialError::TopLevelWgslUnparseable)?
                    .len_utf8()
            }
            _ => {}
        }
    }
    Err(MaterialError::TopLevelWgslUnparseable)
}

fn next_char(source: &str, position: usize) -> Option<char> {
    source.get(position..)?.chars().next()
}

fn is_whitespace(character: char) -> bool {
    matches!(
        character,
        ' ' | '\t'..='\r' | '\u{0085}' | '\u{200e}' | '\u{200f}' | '\u{2028}' | '\u{2029}'
    )
}

fn is_identifier_start(character: char) -> bool {
    character == '_' || unicode_ident::is_xid_start(character)
}
