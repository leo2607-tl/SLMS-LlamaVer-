from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from pathlib import Path
import sys
from qdrant_client.models import Filter
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

class QdrantViewer:
    def __init__(self, qdrant_client: QdrantClient):
        self.client = qdrant_client
    
    def show_all_points(
        self, 
        collection_name: str,
        limit: int = 100,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[Dict]:
        """
        Lấy toàn bộ points từ Qdrant collection sử dụng scroll API
        
        Args:
            collection_name: Tên collection
            limit: Số lượng points mỗi lần scroll (default: 100)
            with_payload: Có lấy payload không (default: True)
            with_vectors: Có lấy vectors không (default: False)
            
        Returns:
            List các points với đầy đủ thông tin
        """
        all_points = []
        offset = None
        
        while True:
            response = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            points, next_offset = response
            
            all_points.extend(points)
            
            if next_offset is None:
                break
                
            offset = next_offset
        
        return all_points
    
    def display_points(
        self, 
        collection_name: str,
        max_display: Optional[int] = None,
        show_vectors: bool = False
    ):
        """
        Hiển thị thông tin chi tiết của tất cả points
        
        Args:
            collection_name: Tên collection
            max_display: Số lượng points tối đa hiển thị (None = tất cả)
            show_vectors: Có hiển thị vectors không
        """
        print(f"\n{'='*80}")
        print(f"COLLECTION: {collection_name}")
        print(f"{'='*80}\n")
        
        all_points = self.show_all_points(
            collection_name=collection_name,
            with_vectors=show_vectors
        )
        
        total_count = len(all_points)
        print(f"Tổng số points: {total_count}\n")
        
        display_points = all_points[:max_display] if max_display else all_points
        
        for idx, point in enumerate(display_points, 1):
            print(f"--- Point {idx} ---")
            print(f"ID: {point.id}")
            
            if point.payload:
                print(f"Payload:")
                for key, value in point.payload.items():
                    if isinstance(value, str) and len(value) > 400:
                        print(f"  {key}: {value[:400]}...")
                    else:
                        print(f"  {key}: {value}")
            
            if show_vectors and point.vector:
                vector_preview = point.vector[:10] if len(point.vector) > 10 else point.vector
                print(f"Vector (preview): {vector_preview}... (dim={len(point.vector)})")
            
            print()
        
        if max_display and total_count > max_display:
            print(f"... và {total_count - max_display} points khác")
        
        print(f"{'='*80}\n")
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        Lấy thống kê về collection
        
        Args:
            collection_name: Tên collection
            
        Returns:
            Dict chứa thông tin thống kê
        """
        collection_info = self.client.get_collection(collection_name)
        
        stats = {
            'name': collection_name,
            'points_count': collection_info.points_count,
            'vectors_count': collection_info.vectors_count,
            'indexed_vectors_count': collection_info.indexed_vectors_count,
            'status': collection_info.status,
            'optimizer_status': collection_info.optimizer_status,
        }
        
        return stats
    
    def search_points_by_filter(
        self,
        collection_name: str,
        filter_condition: Filter,
        limit: int = 100
    ) -> List[Dict]:
        """
        Tìm points theo filter (không dùng vector similarity)
        
        Args:
            collection_name: Tên collection
            filter_condition: Điều kiện filter
            limit: Số lượng points tối đa
            
        Returns:
            List các points thỏa mãn điều kiện
        """
        filtered_points = []
        offset = None
        
        while len(filtered_points) < limit:
            response = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=min(100, limit - len(filtered_points)),
                offset=offset,
                with_payload=True
            )
            
            points, next_offset = response
            filtered_points.extend(points)
            
            if next_offset is None:
                break
                
            offset = next_offset
        
        return filtered_points


if __name__ == "__main__":
    from database.qdrant_database import QdrantDB
    from config.config import DatabaseConfig
    
    qr_client = QdrantDB(url=DatabaseConfig.QDRANT_URL)
    viewer = QdrantViewer(qr_client.qr_client)
    
    collection_name = DatabaseConfig.QDRANT_COLLECTION
    
    viewer.display_points(
        collection_name=collection_name,
        max_display=10,
        show_vectors=True
    )
    
    stats = viewer.get_collection_stats(collection_name)
    print("Collection Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    all_points = viewer.show_all_points(
        collection_name=collection_name,
        with_payload=True,
        with_vectors=False
    )
    print(f"\nTổng số points: {len(all_points)}")
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value="some_doc_id")
            )
        ]
    )
    
    filtered = viewer.search_points_by_filter(
        collection_name=collection_name,
        filter_condition=filter_condition
    )
    print(f"Points thỏa mãn filter: {len(filtered)}")
