import io
import sqlite3
from pathlib import Path

import pytest

import translator


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(translator, 'DB_PATH', str(tmp_path / 'translations.db'))
    translator.init_db()

    class AvailableOllama:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        translator.requests, 'get', lambda *_args, **_kwargs: AvailableOllama(),
    )
    translator.app.config.update(TESTING=True)
    return translator.app.test_client()


def test_merge_places_new_terms_before_divider_and_keeps_shared_precedence():
    merged = translator.merge_shared_glossary(
        'Mina => Mína | exact\nRiver => Río | preferred',
        ' mina  => Mina | inflectable\n# canon\nOld place => Vieja | exact',
    )

    assert merged == (
        'River => Río | preferred\n-----\n'
        ' mina  => Mina | inflectable\n# canon\nOld place => Vieja | exact'
    )


def test_dollar_marks_book_specific_entries_and_hash_remains_a_comment():
    assert translator._glossary_entry('$ Canon => Canon | exact') == ('canon', True)
    assert translator._glossary_entry('# Canon => Canon | exact') is None
    assert translator.glossary_text_for_pipeline('# Canon => Canon | exact') == '# Canon => Canon | exact'


def test_start_file_writer_creates_missing_shared_file_without_divider(tmp_path, monkeypatch):
    monkeypatch.setattr(translator, 'SHARED_GLOSSARIES_FOLDER', str(tmp_path / 'shared'))

    translator.write_shared_glossary('terms', 'New => New\n-----\nOld => Old')

    assert (tmp_path / 'shared' / 'terms.txt').read_text(encoding='utf-8') == (
        'New => New\nOld => Old\n'
    )


@pytest.mark.parametrize('filename', ['../outside.txt', 'nested/name.txt', '..'])
def test_shared_glossary_rejects_names_outside_managed_folder(filename, tmp_path, monkeypatch):
    monkeypatch.setattr(translator, 'SHARED_GLOSSARIES_FOLDER', str(tmp_path / 'shared'))

    with pytest.raises(ValueError, match='inside the shared glossaries folder'):
        translator.shared_glossary_path(filename)


def test_verification_prompt_ignores_ui_divider(client):
    response = client.post('/glossary-verification-prompt', json={
        'glossary': 'New => New\n-----\nOld => Old',
        'sourceLanguage': 'en',
        'targetLanguage': 'ru',
    })

    assert response.status_code == 200
    assert '-----' not in response.get_json()['prompt']


def test_start_does_not_create_translation_if_shared_file_write_fails(
    client, tmp_path, monkeypatch,
):
    monkeypatch.setattr(translator, 'DB_PATH', str(tmp_path / 'translations.db'))
    translator.init_db()
    monkeypatch.setattr(
        translator,
        'read_uploaded_book',
        lambda *_args, **_kwargs: (
            'Source text', None, 'Book', 'Author', 'txt', None,
        ),
    )

    def fail_write(*_args, **_kwargs):
        raise ValueError('Could not write shared glossary: read only')

    monkeypatch.setattr(translator, 'write_shared_glossary', fail_write)
    response = client.post('/translate', data={
        'file': (io.BytesIO(b'Source text'), 'book.txt'),
        'sourceLanguage': 'en',
        'targetLanguage': 'ru',
        'model': 'model',
        'sharedGlossaryEnabled': 'true',
        'sharedGlossaryFilename': 'terms.txt',
        'glossary': 'New => New\n-----\nOld => Old',
    }, content_type='multipart/form-data')

    assert response.status_code == 400
    assert 'Could not write shared glossary' in response.get_json()['error']
    with sqlite3.connect(tmp_path / 'translations.db') as conn:
        assert conn.execute('SELECT COUNT(*) FROM translations').fetchone()[0] == 0


def test_start_writes_combined_glossary_and_creates_missing_file(
    client, tmp_path, monkeypatch,
):
    shared_folder = tmp_path / 'shared'
    monkeypatch.setattr(translator, 'SHARED_GLOSSARIES_FOLDER', str(shared_folder))
    monkeypatch.setattr(
        translator,
        'read_uploaded_book',
        lambda *_args, **_kwargs: (
            'Source text', None, 'Book', 'Author', 'txt', None,
        ),
    )

    class IdleTranslator:
        def __init__(self, *args, **kwargs):
            pass

        def translate_stage1(self, *args, **kwargs):
            return iter(())

    monkeypatch.setattr(translator, 'BookTranslator', IdleTranslator)
    monkeypatch.setattr(translator, '_emit_progress', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translator, '_start_detached_job', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(translator, '_sse_from_progress_queue', lambda *_args, **_kwargs: iter(()))

    response = client.post('/translate', data={
        'file': (io.BytesIO(b'Source text'), 'book.txt'),
        'sourceLanguage': 'en',
        'targetLanguage': 'ru',
        'model': 'model',
        'sharedGlossaryEnabled': 'true',
        'sharedGlossaryFilename': 'terms.txt',
        'glossary': 'New => New\n-----\nOld => Old',
    }, content_type='multipart/form-data')

    assert response.status_code == 200
    assert (shared_folder / 'terms.txt').read_text(encoding='utf-8') == 'New => New\nOld => Old\n'


def test_non_loopback_clients_are_rejected(client):
    response = client.get('/', environ_base={'REMOTE_ADDR': '192.0.2.10'})

    assert response.status_code == 403


def test_cross_origin_pages_are_rejected_even_when_the_browser_is_local(client):
    response = client.get('/', headers={'Origin': 'https://attacker.example'})

    assert response.status_code == 403


def test_completed_downloads_use_original_upload_stem(
    client, tmp_path, monkeypatch,
):
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(translator, 'DB_PATH', str(database_path))
    translator.init_db()
    with sqlite3.connect(database_path) as conn:
        cursor = conn.execute(
            '''INSERT INTO translations (
                filename, source_lang, target_lang, model, status,
                translated_text, source_format
            ) VALUES ('My.Book.epub', 'en', 'ru', 'model', 'completed', 'Done', 'txt')'''
        )
        translation_id = cursor.lastrowid

    response = client.get(f'/download/{translation_id}')

    assert response.status_code == 200
    assert response.headers['Content-Disposition'].endswith('filename=My.Book.txt')


def test_completed_epub_download_uses_original_upload_stem(client, tmp_path, monkeypatch):
    database_path = tmp_path / 'translations.db'
    monkeypatch.setattr(translator, 'DB_PATH', str(database_path))
    translator.init_db()
    with sqlite3.connect(database_path) as conn:
        cursor = conn.execute(
            '''INSERT INTO translations (
                filename, source_lang, target_lang, model, status,
                translated_text, source_format, translated_chapters, book_title, book_author
            ) VALUES ('My.Book.epub', 'en', 'ru', 'model', 'completed', 'Done', 'epub', '["Done"]', 'Book', 'Author')'''
        )
        translation_id = cursor.lastrowid

    response = client.get(f'/download/{translation_id}')

    assert response.status_code == 200
    assert response.headers['Content-Disposition'].endswith('filename=My.Book.epub')


def test_frontend_keeps_target_and_genre_and_uses_original_export_stems():
    page = (Path(__file__).parents[1] / 'src' / 'static' / 'index.html').read_text(encoding='utf-8')

    assert "localStorage.removeItem(WORKSPACE_KEYS.sourceLanguage)" in page
    assert 'restoreWorkspaceFields();' in page[page.index('function resetForm()'):]
    assert 'a.download = `${originalUploadBaseName()}.txt`' in page
    assert 'a.download = `${originalUploadBaseName()}.html`' in page
    assert 'a.download = `${originalUploadBaseName()}.epub`' in page
