from django.shortcuts import render
from board.models import Board

# Create your views here.

def home(request):
    context = {}
    context["title"] = "django homepage"
    return render(request, 'dashboard.html', context)

def board(request):
    rsBoard = Board.objects.all()
    return render(request, 'board_list.html', {'rsBoard' : rsBoard})

#branch-test!