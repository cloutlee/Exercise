import java.util.List;

import redis.clients.jedis.Jedis;

public class Temp {

	public static void main(String[] args) {
		Jedis je = new Jedis("localhost");
		
		System.out.println(je.ping());
		
		je.set("aaa", "abc");
		System.out.println(je.get("aaa"));

		je.lpush("fruitList", "apple");
		je.lpush("fruitList", "orange");
		List<String> list = je.lrange("fruitList", 0, 6);
		for (int i = 0; i < list.size(); i++) {
			System.out.println(list.get(i));
		}
	}
}
