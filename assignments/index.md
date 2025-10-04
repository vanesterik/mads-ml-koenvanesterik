---
layout: page
title: Assignments
permalink: assignments
category: assignments
---

{% for post in site.categories.assignments %}
  <div class="py-1">
    <h3 class="text-black dark:text-white text-lg font-bold"><a href="{{ site.baseurl }}{{ post.url }}" class="hover:text-secondary-500">{{ post.title }}</a></h3>
    <div class="text-sm text-gray-400">{{ post.date | date: "%d %B %Y" }}</div>
  </div>
{% endfor %}