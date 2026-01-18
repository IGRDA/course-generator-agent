"""YouTube video search using yt-dlp for rich metadata extraction."""

from typing import List
import yt_dlp


def search_videos(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for YouTube videos and extract full metadata using yt-dlp.
    
    Args:
        query: Search query for videos
        max_results: Maximum number of videos to return (default: 5)
        
    Returns:
        List of video results with full metadata:
        - title: Video title
        - url: YouTube video URL
        - duration: Duration in seconds
        - published_at: Publication timestamp in milliseconds
        - thumbnail: Thumbnail URL (maxresdefault or best available)
        - channel: Channel name
        - views: View count
        - likes: Like count
    """
    # yt-dlp options for search
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,  # We need full extraction for metadata
        'skip_download': True,
        'ignoreerrors': True,
    }
    
    results = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Use ytsearch to search YouTube
            search_query = f"ytsearch{max_results}:{query}"
            search_result = ydl.extract_info(search_query, download=False)
            
            if not search_result or 'entries' not in search_result:
                return results
            
            for entry in search_result.get('entries', []):
                if entry is None:
                    continue
                
                # Extract video ID
                video_id = entry.get('id', '')
                
                # Get best thumbnail (prefer maxresdefault)
                thumbnail = _get_best_thumbnail(entry, video_id)
                
                # Convert upload_date (YYYYMMDD) to milliseconds timestamp
                published_at = _parse_upload_date(entry.get('upload_date'))
                
                video_data = {
                    'title': entry.get('title', ''),
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'duration': entry.get('duration', 0) or 0,
                    'published_at': published_at,
                    'thumbnail': thumbnail,
                    'channel': entry.get('channel', '') or entry.get('uploader', ''),
                    'views': entry.get('view_count', 0) or 0,
                    'likes': entry.get('like_count', 0) or 0,
                }
                
                results.append(video_data)
                
    except Exception as e:
        return [{"error": f"YouTube search failed: {str(e)}"}]
    
    return results


def _get_best_thumbnail(entry: dict, video_id: str) -> str:
    """
    Get the best available thumbnail URL.
    
    Prefers maxresdefault, falls back to other resolutions.
    """
    thumbnails = entry.get('thumbnails', [])
    
    # Sort by preference (larger is better)
    if thumbnails:
        # Try to find maxresdefault or highest resolution
        sorted_thumbs = sorted(
            thumbnails,
            key=lambda t: (t.get('height', 0) or 0) * (t.get('width', 0) or 0),
            reverse=True
        )
        if sorted_thumbs:
            return sorted_thumbs[0].get('url', '')
    
    # Fallback: construct maxresdefault URL
    if video_id:
        return f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
    
    return entry.get('thumbnail', '')


def _parse_upload_date(upload_date: str | None) -> int:
    """
    Convert YYYYMMDD string to milliseconds timestamp.
    
    Args:
        upload_date: Date string in YYYYMMDD format
        
    Returns:
        Timestamp in milliseconds, or 0 if parsing fails
    """
    if not upload_date or len(upload_date) != 8:
        return 0
    
    try:
        from datetime import datetime
        dt = datetime.strptime(upload_date, '%Y%m%d')
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return 0


if __name__ == "__main__":
    query = "tutorial entrelazamiento cuántico"
    max_results = 3
    
    print(f"🔍 Searching videos for: '{query}'")
    print("-" * 80)
    
    results = search_videos(query, max_results)
    
    print(f"\n🎥 Found {len(results)} YouTube videos:\n")
    for i, video in enumerate(results, 1):
        if "error" in video:
            print(f"❌ {video['error']}")
        else:
            print(f"{i}. {video['title']}")
            print(f"   URL: {video['url']}")
            print(f"   Channel: {video['channel']}")
            print(f"   Duration: {video['duration']}s")
            print(f"   Views: {video['views']:,}")
            print(f"   Likes: {video['likes']:,}")
            print(f"   Thumbnail: {video['thumbnail']}")
            print()

